#!/bin/bash
#SBATCH --job-name=finger_v4_dl
#SBATCH --output=logs/finger_v4_%j.out
#SBATCH --error=logs/finger_v4_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# HPC Innovation Hub - Finger Tapping v4 Deep Learning

echo "======================================"
echo "Finger Tapping v4 Deep Learning"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

# Load modules
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source ~/hawkeye/venv/bin/activate

# Navigate to project
cd ~/hawkeye

# Create directories
mkdir -p logs models results

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run training
python scripts/train_finger_v4_gpu.py \
    --model all \
    --epochs 200 \
    --batch_size 32 \
    --hidden 128

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "======================================"
