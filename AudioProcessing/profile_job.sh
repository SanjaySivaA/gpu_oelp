#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR


module load anaconda3-2024
echo "loaded anaconda"
source /home/apps/compilers/anaconda3/2024/bin/activate
echo "activated anaconda"

conda create --name profile_env python=3.10 -y
conda activate profile_env
echo "created and activated environment"
pip install -r requirements.txt
echo "pip install done"

# 2. PyTorch Native Profiler
# Captures standard op-level traces (saved to ./logs/tensorboard/)
echo "======================================"
echo " Running PyTorch Profiler"
echo "======================================"
python keyword_spotting.py --max-steps 10 --epochs 1

# 3. Nsight Systems (nsys)
# Captures system-wide interactions, CPU/GPU concurrency, and API latency.
echo "======================================"
echo " Running Nsight Systems (nsys)"
echo "======================================"
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=ast_nsys_report \
    --force-overwrite=true \
    python keyword_spotting.py --max-steps 10 --epochs 1

# 4. Nsight Compute (ncu)
# Captures kernel-level architectural metrics (Tensor Core utilization, SM occupancy).
# Note: We limit to 3 steps here because NCU serializes and replays kernels, 
# resulting in massive execution overhead.
echo "======================================"
echo " Running Nsight Compute (ncu)"
echo "======================================"
ncu \
    --set full \
    --output=ast_ncu_report \
    --force-overwrite \
    python keyword_spotting.py --max-steps 3 --epochs 1

echo "Profiling Complete. Reports generated in $SLURM_SUBMIT_DIR"