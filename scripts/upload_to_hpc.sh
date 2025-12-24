#!/bin/bash
# Upload all files to HPC in one command

echo "============================================================"
echo "Uploading Leg Agility files to HPC"
echo "============================================================"

cd /c/Users/YK/tulip/Hawkeye

echo ""
echo "[1/7] Uploading extraction scripts..."
scp scripts/extract_leg_agility_skeletons_v2.py gun3856@10.246.246.111:~/hawkeye/scripts/
scp scripts/extract_leg_agility_skeletons_v3_parallel.py gun3856@10.246.246.111:~/hawkeye/scripts/

echo ""
echo "[2/7] Uploading cleanup script..."
scp scripts/cleanup_old_leg_pkl.sh gun3856@10.246.246.111:~/hawkeye/scripts/

echo ""
echo "[3/7] Uploading verification scripts..."
scp scripts/verify_hpc_structure.sh gun3856@10.246.246.111:~/hawkeye/scripts/
scp scripts/analysis/verify_leg_v2_extraction.py gun3856@10.246.246.111:~/hawkeye/scripts/analysis/

echo ""
echo "[4/7] Uploading train pkl (11MB, ~1 min)..."
scp data/leg_agility_train_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/

echo ""
echo "[5/7] Uploading test pkl (2.4MB, ~10 sec)..."
scp data/leg_agility_test_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/

echo ""
echo "[6/7] Setting permissions on HPC..."
ssh gun3856@10.246.246.111 "chmod +x ~/hawkeye/scripts/cleanup_old_leg_pkl.sh ~/hawkeye/scripts/verify_hpc_structure.sh"

echo ""
echo "[7/7] Cleaning up old pkl files on HPC..."
ssh gun3856@10.246.246.111 "cd ~/hawkeye && rm -f leg_agility_train.pkl leg_agility_test.pkl data/leg_agility_train.pkl data/leg_agility_test.pkl"

echo ""
echo "============================================================"
echo "✅ Upload complete!"
echo "============================================================"
echo ""
echo "Next: Verify on HPC"
echo "  ssh gun3856@10.246.246.111"
echo "  cd ~/hawkeye"
echo "  bash scripts/verify_hpc_structure.sh"
echo "  conda activate hawkeye"
echo "  python scripts/analysis/verify_leg_v2_extraction.py"
