
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_registry import ModelRegistry
from models.ml_wrapper import FingerTappingMLModel
import torch

def verify_gpu():
    print("=== GPU Verification ===")
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA NOT Available. Using CPU.")
        return False

def verify_registry():
    print("\n=== Model Registry Verification ===")
    registry = ModelRegistry()
    
    # Register manually to ensure (simulating app startup)
    registry.register(FingerTappingMLModel("finger_tapping"))
    
    models = registry.get_models("finger_tapping")
    print(f"Registered Models for 'finger_tapping': {len(models)}")
    
    for i, model in enumerate(models):
        print(f"[{i+1}] {model.metadata.name} (v{model.metadata.version}) - Priority: {model.metadata.priority} - Type: {model.metadata.model_type}")
        if model.metadata.model_type == "ml":
            print("   -> Confirmed ML Model presence.")

def main():
    has_gpu = verify_gpu()
    verify_registry()
    
    # If users wants 90% accuracy, they likely need GPU for training/inference speed
    # This verification ensures the infrastructure is ready.
    if has_gpu:
        print("\n[SUCCESS] System accepts GPU operations.")
    else:
        print("\n[WARNING] GPU not detected. ML models will run on CPU (slower).")

if __name__ == "__main__":
    main()
