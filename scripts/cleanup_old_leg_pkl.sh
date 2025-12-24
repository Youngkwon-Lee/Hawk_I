#!/bin/bash
# Cleanup old Leg Agility pkl files (Local + HPC)

echo "=========================================="
echo "Leg Agility PKL Cleanup"
echo "=========================================="

# Local cleanup
echo ""
echo "[LOCAL] Deleting old pkl files..."
LOCAL_DIR="C:/Users/YK/tulip/Hawkeye/data"

if [ -f "$LOCAL_DIR/leg_agility_train.pkl" ]; then
    rm "$LOCAL_DIR/leg_agility_train.pkl"
    echo "  Deleted: leg_agility_train.pkl"
fi

if [ -f "$LOCAL_DIR/leg_agility_test.pkl" ]; then
    rm "$LOCAL_DIR/leg_agility_test.pkl"
    echo "  Deleted: leg_agility_test.pkl"
fi

echo ""
echo "[LOCAL] Cleanup complete"

# HPC cleanup instructions
echo ""
echo "=========================================="
echo "HPC Cleanup Instructions"
echo "=========================================="
echo ""
echo "1. SSH to HPC:"
echo "   ssh your_username@hpc_address"
echo ""
echo "2. Navigate to Hawkeye directory:"
echo "   cd ~/hawkeye"
echo ""
echo "3. Delete old pkl files:"
echo "   rm -f leg_agility_train.pkl"
echo "   rm -f leg_agility_test.pkl"
echo "   rm -f data/leg_agility_train.pkl"
echo "   rm -f data/leg_agility_test.pkl"
echo ""
echo "4. Verify deletion:"
echo "   ls -lh *.pkl"
echo "   ls -lh data/*.pkl"
echo ""
echo "5. (Optional) If stored in ~/hawkeye/data/:"
echo "   find ~/hawkeye -name 'leg_agility*.pkl' -delete"
echo ""
echo "=========================================="
echo "After cleanup, upload new v2 pkl files:"
echo "=========================================="
echo ""
echo "From local (Windows):"
echo "  scp C:/Users/YK/tulip/Hawkeye/data/leg_agility_train_v2.pkl your_username@hpc:~/hawkeye/data/"
echo "  scp C:/Users/YK/tulip/Hawkeye/data/leg_agility_test_v2.pkl your_username@hpc:~/hawkeye/data/"
echo ""
echo "Or use WinSCP / FileZilla for GUI upload"
echo ""
