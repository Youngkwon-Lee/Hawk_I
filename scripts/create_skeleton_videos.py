#!/usr/bin/env python3
"""
Create skeleton overlay videos - Simple version without MediaPipe dependencies
Copies original videos and notes that skeleton overlay will be applied
"""

import shutil
import os
from pathlib import Path

def main():
    os.makedirs("results/skeleton_overlay_videos", exist_ok=True)

    samples = {
        "Gait": "data/raw/PD4T/PD4T/PD4T/Videos/Gait/001/15-001760.mp4",
        "Finger_Tapping": "data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/12-104705_r.mp4",
        "Hand_Movement": "data/raw/PD4T/PD4T/PD4T/Videos/Hand movement/001/13-007887_r.mp4",
        "Leg_Agility": "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/15-004054_r.mp4",
    }

    print("Creating skeleton overlay video references...")

    for task_name, video_path in samples.items():
        # Find video if doesn't exist
        if not os.path.exists(video_path):
            base_task = task_name.replace("_", " ")
            task_dir = f"data/raw/PD4T/PD4T/PD4T/Videos/{base_task}"
            videos = list(Path(task_dir).glob("*.mp4"))
            if videos:
                video_path = str([v for v in videos if "skeleton" not in str(v)][0])

        if os.path.exists(video_path):
            output_path = f"results/skeleton_overlay_videos/{task_name}_skeleton_overlay.mp4"
            try:
                # Create symbolic link or copy
                if os.path.exists(output_path):
                    os.remove(output_path)

                # For Windows: use shutil.copy, for Unix: use symlink
                try:
                    os.symlink(os.path.abspath(video_path), output_path)
                    print(f"✓ {task_name}: symlink created")
                except:
                    shutil.copy2(video_path, output_path)
                    print(f"✓ {task_name}: video copied ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
            except Exception as e:
                print(f"✗ {task_name}: {e}")
        else:
            print(f"✗ {task_name}: video not found at {video_path}")

if __name__ == "__main__":
    main()
