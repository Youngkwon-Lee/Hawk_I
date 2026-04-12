#!/bin/bash
echo "============================================================"
echo "Mamba + CORAL Ordinal - Hand Movement Task"
echo "Architecture: Mamba (Temporal) → CORAL Ordinal Regression"
echo "Baseline test for comparison with ActionMamba"
echo "============================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hawkeye
export CUDA_VISIBLE_DEVICES=0

cd ~/hawkeye
python scripts/train_hand_ordinal.py \
    --epochs 200 \
    --batch_size 32 \
    2>&1 | tee results/hand_ordinal_$(date +%Y%m%d_%H%M%S).log

echo "Hand Movement CORAL training complete!"
