"""
MediaPipe Processor for Skeleton Extraction
Extracts hand landmarks (finger tapping) and pose landmarks (gait)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class LandmarkFrame:
    """Single frame's landmark data"""
    frame_number: int
    landmarks: List[Dict[str, float]]  # [{"id": int, "x": float, "y": float, "z": float, "visibility": float}]
    timestamp: float


class MediaPipeProcessor:
    """
    Extract skeleton data using MediaPipe

    Supports:
    - Hand Landmarks (21 points) for finger tapping
    - Pose Landmarks (33 points) for gait analysis
    """

    def __init__(self, mode: str = "hand"):
        """
        Args:
            mode: "hand" for finger tapping, "pose" for gait
        """
        self.mode = mode

        # Initialize MediaPipe
        if mode == "hand":
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        elif mode == "pose":
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'hand' or 'pose'")

    def process_video(
        self,
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        output_video_path: Optional[str] = None
    ) -> List[LandmarkFrame]:
        """
        Process video and extract landmarks for all frames

        Args:
            video_path: Path to video file
            roi: Optional ROI (x, y, w, h) to crop frames
            output_video_path: Optional path to save video with skeleton overlay

        Returns:
            List of LandmarkFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        landmark_frames = []

        # Initialize video writer if output path is provided
        video_writer = None
        if output_video_path:
            # Use mp4v codec first (avoids OpenH264 issues), then convert to H.264 with ffmpeg
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # # Use avc1 (H.264) codec for better browser compatibility
            # fourcc = cv2.VideoWriter_fourcc(*'avc1')
            output_width = roi[2] if roi else frame_width
            output_height = roi[3] if roi else frame_height
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (output_width, output_height)
            )
            print(f"💾 Saving skeleton overlay video to: {output_video_path}")
            if not video_writer.isOpened():
                print(f"❌ Error: Could not initialize VideoWriter with mp4v codec")
                video_writer = None

        print(f"\n{'='*50}")
        print(f"MediaPipe Processing ({self.mode} mode)")
        print(f"{'='*50}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply ROI if provided
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            landmarks = self._process_frame(frame_rgb)

            if landmarks:
                timestamp = frame_count / fps
                landmark_frame = LandmarkFrame(
                    frame_number=frame_count,
                    landmarks=landmarks,
                    timestamp=timestamp
                )
                landmark_frames.append(landmark_frame)

                # Draw skeleton on frame if video writer is enabled
                if video_writer:
                    frame_with_skeleton = self._draw_skeleton(frame.copy(), frame_rgb)
                    video_writer.write(frame_with_skeleton)
            else:
                # Write original frame if no landmarks detected
                if video_writer:
                    video_writer.write(frame)

            frame_count += 1

            # Progress indicator
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames ({len(landmark_frames)} with landmarks)...")

        cap.release()
        if video_writer:
            video_writer.release()
            print(f"✅ Skeleton overlay video saved!")

            # Convert to H.264 for browser compatibility
            if output_video_path and os.path.exists(output_video_path):
                try:
                    import subprocess
                    temp_path = output_video_path.replace('.mp4', '_temp.mp4')

                    # Convert using ffmpeg
                    cmd = [
                        'ffmpeg', '-i', output_video_path,
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-y',
                        temp_path
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0 and os.path.exists(temp_path):
                        # Replace original with H.264 version
                        os.replace(temp_path, output_video_path)
                        print(f"✅ Converted to H.264 for browser compatibility")
                    else:
                        print(f"⚠️ ffmpeg conversion failed, keeping original")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                except Exception as e:
                    print(f"⚠️ Video conversion error: {e}")

        print(f"\nTotal frames: {frame_count}")
        print(f"Frames with landmarks: {len(landmark_frames)}")
        print(f"Detection rate: {len(landmark_frames)/frame_count*100:.1f}%\n")

        return landmark_frames

    def _process_frame(self, frame_rgb: np.ndarray) -> Optional[List[Dict[str, float]]]:
        """
        Process single frame and extract landmarks

        Returns:
            List of landmark dicts or None if no detection
        """
        if self.mode == "hand":
            return self._process_hand_frame(frame_rgb)
        elif self.mode == "pose":
            return self._process_pose_frame(frame_rgb)

    def _draw_skeleton(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Draw skeleton overlay on frame

        Args:
            frame_bgr: Original frame in BGR format
            frame_rgb: Frame in RGB format for MediaPipe

        Returns:
            Frame with skeleton overlay
        """
        if self.mode == "hand":
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        elif self.mode == "pose":
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

        return frame_bgr

    def _process_hand_frame(self, frame_rgb: np.ndarray) -> Optional[List[Dict[str, float]]]:
        """Extract hand landmarks (21 points per hand)"""
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        # Get first detected hand (or combine both hands)
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": 1.0  # Hand landmarks don't have visibility
            })

        return landmarks

    def _process_pose_frame(self, frame_rgb: np.ndarray) -> Optional[List[Dict[str, float]]]:
        """Extract pose landmarks (33 points)"""
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })

        return landmarks

    def save_to_json(self, landmark_frames: List[LandmarkFrame], output_path: str):
        """Save landmark data to JSON file"""
        import json

        data = []
        for lf in landmark_frames:
            data.append({
                "frame": lf.frame_number,
                "timestamp": lf.timestamp,
                "keypoints": lf.landmarks
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(data)} frames to {output_path}")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python mediapipe_processor.py <video_path> <mode>")
        print("  mode: 'hand' or 'pose'")
        sys.exit(1)

    video_path = sys.argv[1]
    mode = sys.argv[2]

    # Process video
    processor = MediaPipeProcessor(mode=mode)
    landmark_frames = processor.process_video(video_path)

    # Save to JSON
    output_path = video_path.replace('.mp4', f'_{mode}_skeleton.json')
    processor.save_to_json(landmark_frames, output_path)

    print(f"\n✅ Done! Skeleton data saved to {output_path}")
