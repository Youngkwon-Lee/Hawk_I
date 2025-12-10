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
    world_landmarks: Optional[List[Dict[str, float]]] = None # Real-world 3D coordinates (meters)


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
        output_video_path: Optional[str] = None,
        frame_skip: int = 1,
        skip_video_generation: bool = False,
        max_duration: Optional[float] = None,
        resize_width: Optional[int] = 720
    ) -> List[LandmarkFrame]:
        """
        Process video and extract landmarks for all frames

        Args:
            video_path: Path to video file
            roi: Optional ROI (x, y, w, h) to crop frames
            output_video_path: Optional path to save video with skeleton overlay
            frame_skip: Process every Nth frame (1=all frames, 2=every other, etc.)
            skip_video_generation: Skip skeleton video generation for faster processing
            max_duration: Maximum duration to process in seconds (None = entire video)
            resize_width: Resize frames to this width for faster processing (None = no resize)
                         Default 720px for ~2x speedup with minimal accuracy loss

        Returns:
            List of LandmarkFrame objects
        """
        import time
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        landmark_frames = []

        # Calculate resize dimensions (maintain aspect ratio)
        resize_scale = 1.0
        if resize_width and frame_width > resize_width:
            resize_scale = resize_width / frame_width
            resized_height = int(frame_height * resize_scale)
            print(f"  [MediaPipe] Resolution: {frame_width}x{frame_height} → {resize_width}x{resized_height} (scale: {resize_scale:.2f})")
        else:
            resize_width = frame_width
            resized_height = frame_height

        # Initialize video writer if output path is provided
        video_writer = None
        if output_video_path and not skip_video_generation:
            # Use mp4v codec first (avoids OpenH264 issues), then convert to H.264 with ffmpeg
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Output at resized resolution
            output_width = roi[2] if roi else resize_width
            output_height = roi[3] if roi else resized_height
            if roi and resize_scale != 1.0:
                output_width = int(roi[2] * resize_scale)
                output_height = int(roi[3] * resize_scale)
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (output_width, output_height)
            )
            print(f"  [MediaPipe] Saving skeleton video to: {os.path.basename(output_video_path)}")
            if not video_writer.isOpened():
                print(f"  [MediaPipe] ERROR: VideoWriter failed to open")
                video_writer = None
        elif skip_video_generation:
            print(f"  [MediaPipe] Skeleton video generation SKIPPED (fast mode)")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  [MediaPipe] Processing {total_frames} frames ({self.mode} mode, skip={frame_skip})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check max_duration limit (stop after N seconds)
            if max_duration and fps > 0 and (frame_count / fps) >= max_duration:
                print(f"  [MediaPipe] Reached max_duration ({max_duration}s) at frame {frame_count}, stopping")
                break

            # Apply ROI if provided
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            # Resize frame for faster processing (if resize_scale != 1.0)
            if resize_scale != 1.0:
                frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

            # Frame sampling: only process MediaPipe on selected frames
            should_process_mediapipe = (frame_count % frame_skip == 0)

            if should_process_mediapipe:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                result = self._process_frame(frame_rgb)

                # _process_frame now returns a tuple (landmarks, world_landmarks) or None
                if result:
                    landmarks, world_landmarks = result
                    timestamp = frame_count / fps
                    landmark_frame = LandmarkFrame(
                        frame_number=frame_count,
                        landmarks=landmarks,
                        world_landmarks=world_landmarks,
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
            else:
                # For skipped frames, just write original to video
                if video_writer:
                    video_writer.write(frame)

            frame_count += 1

            # Progress indicator (every 60 frames)
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"  [MediaPipe] {frame_count}/{total_frames} ({pct:.0f}%) - {len(landmark_frames)} landmarks - {elapsed:.1f}s")

        cap.release()
        if video_writer:
            video_writer.release()
            print(f"[OK] Skeleton overlay video saved!")

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
                        print(f"[OK] Converted to H.264 for browser compatibility")
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

    def _process_frame(self, frame_rgb: np.ndarray) -> Optional[Tuple[List[Dict[str, float]], Optional[List[Dict[str, float]]]]]:
        """
        Process single frame and extract landmarks

        Returns:
            Tuple(landmarks, world_landmarks) or None if no detection
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

    def _process_hand_frame(self, frame_rgb: np.ndarray) -> Optional[Tuple[List[Dict[str, float]], None]]:
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

        # Hand model doesn't support world landmarks in the same way pose does usually
        # or at least we are not prioritizing it now.
        return landmarks, None

    def _process_pose_frame(self, frame_rgb: np.ndarray) -> Optional[Tuple[List[Dict[str, float]], List[Dict[str, float]]]]:
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
            
        world_landmarks = []
        if results.pose_world_landmarks:
            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                world_landmarks.append({
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
        else:
            world_landmarks = None

        return landmarks, world_landmarks

    def save_to_json(self, landmark_frames: List[LandmarkFrame], output_path: str):
        """Save landmark data to JSON file"""
        import json

        data = []
        for lf in landmark_frames:
            data.append({
                "frame": lf.frame_number,
                "timestamp": lf.timestamp,
                "keypoints": lf.landmarks,
                "world_keypoints": lf.world_landmarks
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

    print(f"\n[OK] Done! Skeleton data saved to {output_path}")
