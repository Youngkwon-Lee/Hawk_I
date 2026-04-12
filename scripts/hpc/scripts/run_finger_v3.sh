#!/bin/bash
#SBATCH --job-name=finger_v3
#SBATCH --output=finger_v3_%j.log
#SBATCH --error=finger_v3_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load anaconda3

# Activate conda environment
source activate triage

# Change to working directory
cd /home2/gun3856/hpc

# Run training
echo "Starting Finger v3 Training..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Run all models with weighted sampling and MSE loss
python scripts/train_finger_v3.py 2>&1

# Alternative options:
# python scripts/train_finger_v3.py --loss focal        # Use focal loss
# python scripts/train_finger_v3.py --no-weighting      # Disable weighted sampling
# python scripts/train_finger_v3.py --model attention   # Only attention model
# python scripts/train_finger_v3.py --model feature     # Only feature-aware model
# python scripts/train_finger_v3.py --model advanced    # Only advanced features model

echo "Training completed at: $(date)"
