#!/bin/bash
# HPC Leg Agility Update Script

echo "============================================================"
echo "HPC Leg Agility Data Update"
echo "============================================================"

# Step 1: Update code from git
echo ""
echo "[1/3] Updating code from GitHub..."
cd ~/hawkeye
git pull origin main

# Step 2: Delete old leg agility pkl files
echo ""
echo "[2/3] Deleting old leg agility pkl files..."
rm -f leg_agility_train.pkl leg_agility_test.pkl
rm -f data/leg_agility_train.pkl data/leg_agility_test.pkl
find ~/hawkeye -name 'leg_agility*.pkl' -not -name '*_v2.pkl' -delete

echo "  Old files deleted"

# Step 3: Upload new v2 files (manual step)
echo ""
echo "[3/3] Upload new files from local:"
echo "  scp C:/Users/YK/tulip/Hawkeye/data/leg_agility_train_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/"
echo "  scp C:/Users/YK/tulip/Hawkeye/data/leg_agility_test_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/"

echo ""
echo "============================================================"
echo "HPC update complete!"
echo "============================================================"
