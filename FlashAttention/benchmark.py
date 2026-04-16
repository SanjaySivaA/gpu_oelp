import math
import torch
import custom_flash_attn  # Imports the compiled C++ extension!

def naive_pytorch_fp32(q, k, v, N, d):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.tril(torch.ones(N, N, device='cuda')).view(1, 1, N, N)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)

def tensor_core_pytorch_fp16(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

def benchmark(name, func, *args):
    # Warmup
    for _ in range(5):
        func(*args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(10):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / 10.0
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"[{name}]")
    print(f"  Time:   {avg_time_ms:.3f} ms")
    print(f"  Memory: {peak_memory_mb:.2f} MB\n")

if __name__ == "__main__":
    B = 4
    nh = 12
    N = 1024
    d = 64

    print(f"\nBenchmarking with Tensor Shape: [B={B}, Heads={nh}, Seq={N}, Dim={d}]\n")

    # Prepare FP32 Tensors
    q_fp32 = torch.randn(B, nh, N, d, device='cuda', dtype=torch.float32).contiguous()
    k_fp32 = torch.randn(B, nh, N, d, device='cuda', dtype=torch.float32).contiguous()
    v_fp32 = torch.randn(B, nh, N, d, device='cuda', dtype=torch.float32).contiguous()

    # Prepare FP16 Tensors
    q_fp16 = q_fp32.half().contiguous()
    k_fp16 = k_fp32.half().contiguous()
    v_fp16 = v_fp32.half().contiguous()

    benchmark(
        "1. Naive PyTorch (FP32 - Standard Cores)", 
        naive_pytorch_fp32, q_fp32, k_fp32, v_fp32, N, d
    )
    benchmark(
        "2. PyTorch (FP16 - TENSOR CORES)", 
        tensor_core_pytorch_fp16, q_fp16, k_fp16, v_fp16
    )
    benchmark(
        "3. Custom FlashAttention (FP32)", 
        custom_flash_attn.forward_fp32, q_fp32, k_fp32, v_fp32
    )
    benchmark(
        "4. Custom FlashAttention (FP16 Tensor Cores)", 
        custom_flash_attn.forward_fp16, q_fp16, k_fp16, v_fp16
    )