"""Download BlazePose ONNX models"""

import os
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")

os.makedirs(ONNX_DIR, exist_ok=True)

print("Downloading BlazePose ONNX models...")

# PINTO model zoo - BlazePose Full body
PINTO_BASE = "https://github.com/PINTO0309/PINTO_model_zoo/releases/download/053_BlazePose"

models = [
    ("pose_detection.onnx", f"{PINTO_BASE}/pose_detection.onnx"),
    ("pose_landmark_full.onnx", f"{PINTO_BASE}/pose_landmark_full_body.onnx"),
]

for filename, url in models:
    output_path = os.path.join(ONNX_DIR, filename)
    if os.path.exists(output_path):
        print(f"[OK] Already exists: {filename}")
        continue
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] Downloaded: {filename} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[FAIL] {filename}: {e}")

print(f"\nModels saved to: {ONNX_DIR}")
