"""Debug v3 extraction to find the issue"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Test with one video
VIDEO_PATH = "C:/Users/YK/tulip/Hawkeye/data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/15-001760.mp4"

print("Testing MediaPipe extraction...")
processor = MediaPipeProcessor()
landmarks = processor.process_video(
    VIDEO_PATH,
    task='pose',
    visualization=False,
    verbose=True
)

print(f"\nTotal frames: {len(landmarks)}")
print(f"Type of first element: {type(landmarks[0]) if landmarks else 'Empty'}")

# Check first non-None frame
for i, frame_lm in enumerate(landmarks):
    if frame_lm is not None:
        print(f"\nFrame {i}: {len(frame_lm)} landmarks")
        print(f"Landmark type: {type(frame_lm)}")
        print(f"First landmark: {frame_lm[0]}")

        # Check if landmarks 23-28 exist
        if len(frame_lm) >= 29:
            print("\nLeg landmarks:")
            for idx in [23, 24, 25, 26, 27, 28]:
                lm = frame_lm[idx]
                print(f"  Landmark {idx}: {lm}")
        break
