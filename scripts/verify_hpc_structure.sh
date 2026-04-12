#!/bin/bash
# Verify HPC folder structure matches local

echo "============================================================"
echo "HPC Folder Structure Verification"
echo "============================================================"

echo ""
echo "[1] Checking scripts/extract_leg_agility* files:"
ls -lh ~/hawkeye/scripts/extract_leg_agility*

echo ""
echo "[2] Checking scripts/cleanup script:"
ls -lh ~/hawkeye/scripts/cleanup_old_leg_pkl.sh

echo ""
echo "[3] Checking scripts/analysis/ verification:"
ls -lh ~/hawkeye/scripts/analysis/verify_leg_v2_extraction.py

echo ""
echo "[4] Checking data/ folder:"
ls -lh ~/hawkeye/data/leg_agility*

echo ""
echo "[5] Expected files (should exist):"
echo "  ✓ scripts/extract_leg_agility_skeletons.py (original)"
echo "  ✓ scripts/extract_leg_agility_skeletons_v2.py (NEW)"
echo "  ✓ scripts/extract_leg_agility_skeletons_v3_parallel.py (NEW)"
echo "  ✓ scripts/cleanup_old_leg_pkl.sh (NEW)"
echo "  ✓ scripts/analysis/verify_leg_v2_extraction.py (NEW)"
echo "  ✓ data/leg_agility_train_v2.pkl (UPLOAD NEEDED)"
echo "  ✓ data/leg_agility_test_v2.pkl (UPLOAD NEEDED)"

echo ""
echo "============================================================"
echo "Run this on HPC: bash ~/hawkeye/scripts/verify_hpc_structure.sh"
echo "============================================================"
