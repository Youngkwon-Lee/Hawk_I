"""Verify leg_agility v2 extraction results"""
import pickle
import numpy as np
import os
from pathlib import Path

# Auto-detect base path (works on both Windows and Linux)
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
data_dir = base_dir / 'data'

# Load train
train_path = data_dir / 'leg_agility_train_v2.pkl'
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

# Load test
test_path = data_dir / 'leg_agility_test_v2.pkl'
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

print("="*60)
print("Leg Agility V2 Extraction Verification")
print("="*60)

print("\nTRAIN SET:")
print(f"  X shape: {train_data['X'].shape}")
print(f"  y shape: {train_data['y'].shape}")
print(f"  Scores: {np.bincount(train_data['y'])}")
print(f"  video_id: {len(train_data['video_id'])} videos")
print(f"  side: {len(train_data['side'])} annotations")
print(f"    Left: {train_data['side'].count('left')}")
print(f"    Right: {train_data['side'].count('right')}")

print("\nTEST SET:")
print(f"  X shape: {test_data['X'].shape}")
print(f"  y shape: {test_data['y'].shape}")
print(f"  Scores: {np.bincount(test_data['y'])}")
print(f"  video_id: {len(test_data['video_id'])} videos")
print(f"  side: {len(test_data['side'])} annotations")
print(f"    Left: {test_data['side'].count('left')}")
print(f"    Right: {test_data['side'].count('right')}")

print("\n" + "="*60)
print("✅ Both files loaded successfully!")
print("✅ video_id and side information present!")
print("="*60)
