#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --output=final_eval_%j.log
#SBATCH --error=final_eval_%j.err
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

# Run final evaluation
echo "Starting Final Test Set Evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Run all models (Gait + Finger)
python scripts/evaluate_final.py --task all 2>&1

# Or run specific task:
# python scripts/evaluate_final.py --task gait
# python scripts/evaluate_final.py --task finger

echo "Evaluation completed at: $(date)"
