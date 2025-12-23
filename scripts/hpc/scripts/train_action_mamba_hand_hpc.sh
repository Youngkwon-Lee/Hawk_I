#!/bin/bash
#
# HPC Script: ActionMamba (Mamba + GCN) Training for Hand Movement
# Expected: Pearson 0.70+ (vs baseline)
#

echo "============================================================"
echo "ActionMamba Training - Hand Movement Task"
echo "Architecture: ACE → [GCN ⊕ Mamba] → Fusion → CORAL"
echo "Data: 21 hand landmarks (wrist + 5 fingers)"
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

python scripts/train_action_mamba_hand.py \
    --epochs 200 \
    --batch_size 32 \
    2>&1 | tee results/action_mamba_hand_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
