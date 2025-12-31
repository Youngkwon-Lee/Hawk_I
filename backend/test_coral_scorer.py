"""
Test CORAL Scorer - Verify model loading and inference
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

def test_coral_scorer():
    print("="*60)
    print("CORAL SCORER TEST")
    print("="*60)

    # Import
    try:
        from models.coral_scorer import get_coral_scorer, TORCH_AVAILABLE
        print(f"[OK] Import successful")
        print(f"[OK] PyTorch available: {TORCH_AVAILABLE}")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

    if not TORCH_AVAILABLE:
        print("[SKIP] PyTorch not available")
        return False

    # Load models
    scorer = get_coral_scorer()
    loaded = scorer.load_models()

    if not loaded:
        print("[FAIL] No models loaded")
        return False

    print(f"[OK] Models loaded: {scorer.get_available_tasks()}")

    # Test each task with dummy data
    test_configs = {
        'gait': {'frames': 300, 'features': 30},
        'finger_tapping': {'frames': 150, 'features': 123},
        'hand_movement': {'frames': 150, 'features': 63},
        'leg_agility': {'frames': 150, 'features': 18},
    }

    print("\n" + "="*60)
    print("INFERENCE TEST (Dummy Data)")
    print("="*60)

    results = {}
    for task, config in test_configs.items():
        if task not in scorer.get_available_tasks():
            print(f"[SKIP] {task}: Model not loaded")
            continue

        # Create dummy skeleton sequence
        dummy_skeleton = np.random.randn(config['frames'], config['features']).astype(np.float32)

        try:
            result = scorer.predict(dummy_skeleton, task)
            if result:
                print(f"[OK] {task}: Score={result.score}, Conf={result.confidence:.3f}, Expected={result.expected_score:.3f}")
                results[task] = result
            else:
                print(f"[FAIL] {task}: No result")
        except Exception as e:
            print(f"[FAIL] {task}: {e}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tasks tested: {len(results)}/{len(test_configs)}")

    if len(results) == len(test_configs):
        print("[OK] All tasks working!")
        return True
    else:
        print("[WARN] Some tasks failed")
        return False


if __name__ == "__main__":
    success = test_coral_scorer()
    sys.exit(0 if success else 1)
