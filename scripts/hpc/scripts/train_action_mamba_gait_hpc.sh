#!/bin/bash
#
# HPC Script: ActionMamba (Mamba + GCN) Training for Gait
# Expected: Pearson 0.85+ (vs 0.807 current)
#

echo "============================================================"
echo "ActionMamba Training - Gait Task"
echo "Architecture: ACE → [GCN ⊕ Mamba] → Fusion → CORAL"
echo "Expected: Pearson 0.85+ (+5.3% vs Mamba-only)"
echo "============================================================"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hawkeye

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Run training
cd ~/hawkeye

echo ""
echo "Starting ActionMamba training..."
echo "Using GPU 0 (Tesla V100)"
echo ""

python scripts/train_action_mamba_gait.py \
    --epochs 200 \
    --batch_size 32 \
    2>&1 | tee results/action_mamba_gait_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
