#!/bin/bash
#SBATCH --job-name=stgcn_pd4t
#SBATCH --output=logs/stgcn_%j.out
#SBATCH --error=logs/stgcn_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ST-GCN for PD4T - HPC Job Script

echo "======================================"
echo "ST-GCN for PD4T"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate triage

# Navigate to project
cd ~/hawkeye

# Create directories
mkdir -p logs results

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run ST-GCN training with LOSO CV
python scripts/train_stgcn_gpu.py \
    --task both \
    --epochs 100 \
    --batch_size 32 \
    --hidden 64 \
    --cv loso

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "======================================"
