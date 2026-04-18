import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import os
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.profiler

# --- 1. Arguments & Setup ---
parser = argparse.ArgumentParser(description="AST Profiling Script")
parser.add_argument("--max-steps", type=int, default=0, help="Limit steps per epoch for profiling overhead")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run")
args = parser.parse_args()

os.makedirs('./data', exist_ok=True)
os.makedirs('./logs/tensorboard', exist_ok=True)

print("Downloading/Loading Speech Commands Dataset...")
dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',
    url='speech_commands_v0.02',
    folder_in_archive='SpeechCommands',
    download=True
)

waveform, sample_rate, label, speaker_id, utterance_number = dataset[0]

# --- 2. Audio to Image (Mel-Spectrogram) ---
spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
db_transform = torchaudio.transforms.AmplitudeToDB()
spec = spectrogram_transform(waveform)
spec_db = db_transform(spec).unsqueeze(0)

# --- 3. The Patcher (Linear Projection) ---
patch_size = 16
embed_dim = 128
patcher = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

# Visualization (Saved to disk instead of blocking execution)
plt.figure(figsize=(10, 4))
plt.imshow(spec_db[0, 0].detach().numpy(), cmap='magma', origin='lower', aspect='auto')
plt.title(f"Mel-Spectrogram for Spoken Word: '{label.upper()}'")
plt.ylabel("Frequency (Mel bins)")
plt.xlabel("Time (Frames)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig("spectrogram.png")
plt.close()

# --- 4. Model Architecture ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x

class LightweightAST(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=3, num_classes=35, seq_length=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x[:, 0])
        logits = self.classification_head(x)
        return logits

# --- 5. Data Preparation & Batching ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on Device: {device.type.upper()}")

labels = sorted(list(set(datapoint[2] for datapoint in dataset)))
label_to_index = {label: i for i, label in enumerate(labels)}

def collate_fn(batch):
    tensors, targets = [], []
    for wave, _, lbl, _, _ in batch:
        if wave.shape[1] < 16000:
            pad_amount = 16000 - wave.shape[1]
            wave = nn.functional.pad(wave, (0, pad_amount))
        s = spectrogram_transform(wave)
        s_db = db_transform(s).unsqueeze(0)
        p = patcher(s_db).flatten(2).transpose(1, 2).squeeze(0)
        tensors.append(p)
        targets.append(label_to_index[lbl])
    return torch.stack(tensors), torch.tensor(targets)

subset_indices = list(range(0, 1000))
train_subset = Subset(dataset, subset_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# --- 6. Training Setup ---
model = LightweightAST(embed_dim=128, num_heads=4, num_layers=3, num_classes=35).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda')

# --- 7. The Profiling & Training Loop ---
print("\nStarting Mixed Precision Training (FP16/FP32)...")

for epoch in range(args.epochs):
    model.train()
    total_loss, correct = 0, 0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{args.epochs}")

    # Initialize PyTorch Profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/tensorboard'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, (inputs, targets) in enumerate(loop):
            if args.max_steps > 0 and step >= args.max_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            loop.set_postfix(loss=loss.item())
            
            # Step the profiler forward
            prof.step()

    epoch_acc = (correct / len(train_subset)) * 100
    print(f"-> Epoch {epoch+1} Completed | Loss: {total_loss/max(1, step)} | Accuracy: {epoch_acc:.2f}%")

torch.save(model.state_dict(), "fp32_baseline_model.pth")