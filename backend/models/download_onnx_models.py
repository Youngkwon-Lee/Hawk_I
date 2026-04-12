"""Download BlazePose ONNX models from PINTO model zoo"""

import os
import urllib.request
import tarfile
import shutil

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")

os.makedirs(ONNX_DIR, exist_ok=True)

print("Downloading BlazePose ONNX models from PINTO model zoo...")

# PINTO model zoo - BlazePose archive
ARCHIVE_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/053_BlazePose/resources.tar.gz"
ARCHIVE_PATH = os.path.join(ONNX_DIR, "resources.tar.gz")
EXTRACT_DIR = os.path.join(ONNX_DIR, "temp_extract")

# Check if models already exist
if os.path.exists(os.path.join(ONNX_DIR, "pose_detection.onnx")):
    print("[OK] Models already exist!")
else:
    try:
        # Download archive
        print(f"Downloading resources.tar.gz (this may take a while)...")
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)
        size_mb = os.path.getsize(ARCHIVE_PATH) / (1024 * 1024)
        print(f"[OK] Downloaded: resources.tar.gz ({size_mb:.2f} MB)")

        # Extract archive
        print("Extracting models...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)

        # Find and copy ONNX files
        onnx_files_found = []
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for file in files:
                if file.endswith('.onnx'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(ONNX_DIR, file)
                    shutil.copy2(src_path, dst_path)
                    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
                    print(f"  [OK] {file} ({size_mb:.2f} MB)")
                    onnx_files_found.append(file)

        # Cleanup
        os.remove(ARCHIVE_PATH)
        shutil.rmtree(EXTRACT_DIR)

        print(f"\n[SUCCESS] Found {len(onnx_files_found)} ONNX models")

    except Exception as e:
        print(f"[ERROR] {e}")

        # Cleanup on error
        if os.path.exists(ARCHIVE_PATH):
            os.remove(ARCHIVE_PATH)
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)

print(f"\nModels directory: {ONNX_DIR}")

# List final models
print("\nAvailable ONNX models:")
for f in os.listdir(ONNX_DIR):
    if f.endswith('.onnx'):
        size_mb = os.path.getsize(os.path.join(ONNX_DIR, f)) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
