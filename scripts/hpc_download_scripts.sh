#!/bin/bash
# Download scripts from GitHub to HPC (run this ON HPC)

echo "============================================================"
echo "Downloading Leg Agility scripts from GitHub to HPC"
echo "============================================================"

cd ~/hawkeye

echo ""
echo "[1/5] Downloading v2 extraction script..."
wget -O scripts/extract_leg_agility_skeletons_v2.py \
  "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/extract_leg_agility_skeletons_v2.py"

echo ""
echo "[2/5] Downloading v3 parallel script..."
wget -O scripts/extract_leg_agility_skeletons_v3_parallel.py \
  "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/extract_leg_agility_skeletons_v3_parallel.py"

echo ""
echo "[3/5] Downloading cleanup script..."
wget -O scripts/cleanup_old_leg_pkl.sh \
  "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/cleanup_old_leg_pkl.sh"

echo ""
echo "[4/5] Downloading verification script (Python)..."
mkdir -p scripts/analysis
wget -O scripts/analysis/verify_leg_v2_extraction.py \
  "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/analysis/verify_leg_v2_extraction.py"

echo ""
echo "[5/5] Downloading verification script (Shell)..."
wget -O scripts/verify_hpc_structure.sh \
  "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/verify_hpc_structure.sh"

echo ""
echo "Setting permissions..."
chmod +x scripts/cleanup_old_leg_pkl.sh scripts/verify_hpc_structure.sh

echo ""
echo "Cleaning old pkl files..."
rm -f leg_agility_train.pkl leg_agility_test.pkl
rm -f data/leg_agility_train.pkl data/leg_agility_test.pkl

echo ""
echo "============================================================"
echo "✅ Scripts downloaded!"
echo "============================================================"
echo ""
echo "Next: Upload pkl files from local machine"
echo ""
echo "Run this on LOCAL Git Bash (not HPC):"
echo "  cd /c/Users/YK/tulip/Hawkeye"
echo "  scp data/leg_agility_train_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/"
echo "  scp data/leg_agility_test_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/"
