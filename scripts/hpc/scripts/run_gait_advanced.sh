#!/bin/bash
#SBATCH --job-name=gait_adv
#SBATCH --output=gait_advanced_%j.log
#SBATCH --error=gait_advanced_%j.err
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
echo "Starting Gait Advanced Training..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Run all models
python scripts/train_gait_advanced.py 2>&1

# Or run specific model:
# python scripts/train_gait_advanced.py --model tcn
# python scripts/train_gait_advanced.py --model gcn
# python scripts/train_gait_advanced.py --model dilated

echo "Training completed at: $(date)"
