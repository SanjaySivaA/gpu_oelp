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

module load cuda-12.4

conda activate profile_env
echo "activated environment"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --output=ast_nsys_report \
    --force-overwrite=true \
    python keyword_spotting.py --max-steps 10 --epochs 1
# 4. Nsight Compute (ncu)
# Captures kernel-level architectural metrics (Tensor Core utilization, SM occupancy).
# Note: We limit to 3 steps here because NCU serializes and replays kernels, 
# resulting in massive execution overhead.

echo "Profiling Complete. Reports generated in $SLURM_SUBMIT_DIR"
