"""
Compare Score 2 vs Score 3 Gait Videos with Skeleton Overlay
"""
import cv2
import mediapipe as mp
import numpy as np
import os

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Videos to compare
VIDEOS = {
    'Score 2': {
        'path': f"{PD4T_ROOT}/026/14-001917.mp4",
        'subject': 26,
        'metrics': {'arm_swing': 0.047, 'speed': 0.533, 'cadence': 124.5}
    },
    'Score 3': {
        'path': f"{PD4T_ROOT}/004/13-007586.mp4",
        'subject': 4,
        'metrics': {'arm_swing': 0.148, 'speed': 0.393, 'cadence': 107.8}
    }
}

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_video(video_path, label, metrics):
    """Process video with skeleton overlay"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    max_frames = int(fps * 10)  # 10 seconds max

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw skeleton
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Add label and metrics overlay
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Subject {metrics['subject']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Metrics
        cv2.putText(frame, f"Arm Swing: {metrics['arm_swing']:.3f}", (width-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Speed: {metrics['speed']:.3f}", (width-300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Cadence: {metrics['cadence']:.1f}", (width-300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        frames.append(frame)
        frame_count += 1

    cap.release()
    pose.close()

    return frames, fps, (width, height)

def create_comparison_video():
    """Create side-by-side comparison video"""
    print("Processing Score 2 video...")
    frames2, fps2, size2 = process_video(
        VIDEOS['Score 2']['path'],
        'Score 2 (Mild)',
        {**VIDEOS['Score 2']['metrics'], 'subject': VIDEOS['Score 2']['subject']}
    )

    print("Processing Score 3 video...")
    frames3, fps3, size3 = process_video(
        VIDEOS['Score 3']['path'],
        'Score 3 (Moderate)',
        {**VIDEOS['Score 3']['metrics'], 'subject': VIDEOS['Score 3']['subject']}
    )

    if frames2 is None or frames3 is None:
        print("Failed to process videos")
        return

    # Match frame counts
    min_frames = min(len(frames2), len(frames3))
    frames2 = frames2[:min_frames]
    frames3 = frames3[:min_frames]

    # Resize to same height
    target_height = 480

    def resize_frame(frame, target_h):
        h, w = frame.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        return cv2.resize(frame, (new_w, target_h))

    # Create comparison video
    output_path = f"{OUTPUT_DIR}/score2_vs_score3_comparison.mp4"

    # Get dimensions after resize
    sample2 = resize_frame(frames2[0], target_height)
    sample3 = resize_frame(frames3[0], target_height)
    combined_width = sample2.shape[1] + sample3.shape[1] + 10  # 10px gap

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps2, (combined_width, target_height))

    print(f"Creating comparison video ({min_frames} frames)...")

    for i, (f2, f3) in enumerate(zip(frames2, frames3)):
        # Resize
        f2_resized = resize_frame(f2, target_height)
        f3_resized = resize_frame(f3, target_height)

        # Create combined frame with gap
        gap = np.zeros((target_height, 10, 3), dtype=np.uint8)
        combined = np.hstack([f2_resized, gap, f3_resized])

        out.write(combined)

        if i % 100 == 0:
            print(f"  Frame {i}/{min_frames}")

    out.release()
    print(f"\nSaved: {output_path}")

    # Also save individual videos
    for label, info in VIDEOS.items():
        frames = frames2 if 'Score 2' in label else frames3
        individual_path = f"{OUTPUT_DIR}/{label.replace(' ', '_').lower()}_overlay.mp4"

        sample = resize_frame(frames[0], target_height)
        out = cv2.VideoWriter(individual_path, fourcc, fps2, (sample.shape[1], target_height))

        for f in frames:
            out.write(resize_frame(f, target_height))
        out.release()
        print(f"Saved: {individual_path}")

if __name__ == "__main__":
    create_comparison_video()
