"""
Movement-Based ROI Detection Module
100% Algorithm-based (no ML/DL models)
Ported from example/ex2/src/core/hybrid_vlm_classifier.py
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ROIResult:
    """ROI Detection Result"""
    roi: Tuple[int, int, int, int]  # (x, y, w, h)
    movement_map: np.ndarray
    motion_area_ratio: float
    motion_pattern: str  # 'global' or 'localized'
    body_part: str  # 'hand', 'foot', 'fullbody', 'unknown'
    confidence: float


class MovementBasedROI:
    """
    Movement-based ROI Detector using pure computer vision algorithms
    - Frame differencing for motion detection
    - MediaPipe for body part detection
    - Contour analysis for ROI extraction
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

    def detect(self, video_path: str, num_frames: int = 30) -> ROIResult:
        """
        Detect ROI from video using movement-based approach

        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze for movement map

        Returns:
            ROIResult with ROI coordinates, movement map, and metadata
        """
        # Step 1: Calculate movement map
        print("Step 1: Calculating movement map...")
        movement_map = self._calculate_movement_map(video_path, num_frames)

        # Analyze motion pattern
        adaptive_threshold = 0.05 if movement_map.mean() < 0.02 else 0.3
        motion_mask = (movement_map > adaptive_threshold).astype(np.uint8)
        motion_area_ratio = motion_mask.sum() / (movement_map.shape[0] * movement_map.shape[1])

        # Determine motion pattern
        motion_pattern = "global" if motion_area_ratio > 0.2 else "localized"

        print(f"  Motion pattern: {motion_pattern}")
        print(f"  Motion area ratio: {motion_area_ratio:.1%}")

        # Step 2: Find ROI from movement map
        print("Step 2: Finding ROI from movement...")
        roi = self._find_movement_roi(movement_map, threshold=adaptive_threshold)

        x, y, w, h = roi
        print(f"  ROI: x={x}, y={y}, w={w}, h={h}")

        # Step 3: Detect body part in ROI
        print("Step 3: Detecting body part in ROI...")
        body_part, confidence = self._detect_body_part_in_roi(video_path, roi, movement_map)

        print(f"  Body part: {body_part}")
        print(f"  Confidence: {confidence:.1%}")

        return ROIResult(
            roi=roi,
            movement_map=movement_map,
            motion_area_ratio=motion_area_ratio,
            motion_pattern=motion_pattern,
            body_part=body_part,
            confidence=confidence
        )

    def _calculate_movement_map(self, video_path: str, num_frames: int = 30) -> np.ndarray:
        """
        Calculate temporal movement map from frame differences

        Algorithm:
        1. Sample N frames from video
        2. Convert to grayscale
        3. Calculate absolute difference between consecutive frames
        4. Threshold to binary (threshold=10)
        5. Accumulate and normalize

        Args:
            video_path: Path to video file
            num_frames: Number of frames to process

        Returns:
            Movement map (H x W) with values in [0, 1]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        movement_map = np.zeros((height, width), dtype=np.float32)
        prev_gray = None
        count = 0

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Frame differencing
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                movement_map += thresh.astype(np.float32)
                count += 1

            prev_gray = gray

        cap.release()

        # Normalize
        if count > 0:
            movement_map /= count

        if movement_map.max() > 0:
            movement_map /= movement_map.max()

        return movement_map

    def _find_movement_roi(
        self,
        movement_map: np.ndarray,
        threshold: float = 0.3,
        padding_ratio: float = 0.5
    ) -> Tuple[int, int, int, int]:
        """
        Find ROI from movement map using contour detection

        Algorithm:
        1. Threshold movement map to binary
        2. Apply morphological operations (close + open)
        3. Find contours
        4. Filter noise contours (min area = 0.01% of frame)
        5. Merge multiple contours for full-body motion (gait)
        6. Add padding
        7. Ensure minimum ROI size for MediaPipe

        Args:
            movement_map: Movement map (H x W) in [0, 1]
            threshold: Threshold for binary conversion
            padding_ratio: Padding as ratio of ROI size (0.5 = 50%)

        Returns:
            ROI as (x, y, w, h)
        """
        binary = (movement_map > threshold).astype(np.uint8) * 255

        # Adaptive morphology kernel size
        kernel_size = (5, 5) if threshold < 0.1 else (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"  Found {len(contours)} contours")

        if len(contours) == 0:
            # Default ROI (center of frame)
            h, w = movement_map.shape
            return (w//4, h//4, w//2, h//2)

        # Filter noise contours
        frame_area = movement_map.shape[0] * movement_map.shape[1]
        min_contour_area = frame_area * 0.0001  # 0.01% of frame
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        print(f"  {len(significant_contours)} significant contours")

        if len(significant_contours) == 0:
            h, w = movement_map.shape
            return (w//4, h//4, w//2, h//2)

        # Merge multiple contours for full-body motion (gait)
        if len(significant_contours) > 1:
            all_points = np.vstack([c for c in significant_contours])
            x, y, w, h = cv2.boundingRect(all_points)
        else:
            largest_contour = max(significant_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding
        frame_height, frame_width = movement_map.shape
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

        x_new = max(0, x - pad_w)
        y_new = max(0, y - pad_h)
        w_new = min(frame_width - x_new, w + 2 * pad_w)
        h_new = min(frame_height - y_new, h + 2 * pad_h)

        # Ensure minimum ROI size for MediaPipe
        min_size = 150
        if w_new < min_size:
            diff = min_size - w_new
            x_new = max(0, x_new - diff // 2)
            w_new = min(frame_width - x_new, min_size)
        if h_new < min_size:
            diff = min_size - h_new
            y_new = max(0, y_new - diff // 2)
            h_new = min(frame_height - y_new, min_size)

        return (x_new, y_new, w_new, h_new)

    def _detect_body_part_in_roi(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        movement_map: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """
        Detect which body part is moving in ROI using MediaPipe

        Algorithm:
        1. Extract frames from ROI
        2. Run MediaPipe Hands detection
        3. Run MediaPipe Pose detection
        4. Count hand/foot/fullbody detections
        5. Analyze movement at wrist vs ankle positions
        6. Determine body part based on detection counts

        Args:
            video_path: Path to video file
            roi: ROI coordinates (x, y, w, h)
            movement_map: Optional movement map for movement analysis

        Returns:
            Tuple of (body_part, confidence)
            body_part: 'hand', 'foot', 'fullbody', or 'unknown'
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "unknown", 0.0

        x, y, w, h = roi

        # Extract frames from ROI
        frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            roi_frame = frame[y:y+h, x:x+w]
            frames.append(roi_frame)

        cap.release()

        if len(frames) == 0:
            return "unknown", 0.0

        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose

        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3
        )
        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3
        )

        hand_count = 0
        foot_count = 0
        fullbody_count = 0

        # Process first 5 frames
        for frame in frames[:5]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand detection
            hands_result = hands.process(rgb)
            if hands_result.multi_hand_landmarks:
                hand_count += len(hands_result.multi_hand_landmarks)

            # Pose detection
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks.landmark

                # Check for full body (shoulders + hips + ankles visible)
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
                    left_hip.visibility > 0.5 and right_hip.visibility > 0.5 and
                    left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5):
                    fullbody_count += 1

                # Count foot detections
                if left_ankle.visibility > 0.5:
                    foot_count += 1
                if right_ankle.visibility > 0.5:
                    foot_count += 1

        hands.close()
        pose.close()

        # Determine body part based on counts
        # IMPORTANT: Prioritize fullbody/foot BEFORE hand to avoid misclassifying gait as hand
        total_frames = min(5, len(frames))

        if fullbody_count >= 2:
            # Full body visible (gait detection)
            confidence = fullbody_count / total_frames
            return "fullbody", min(confidence, 1.0)
        elif foot_count >= 3:
            # Feet detected but not full body (leg agility or partial gait)
            confidence = foot_count / (total_frames * 2)  # 2 feet per frame
            return "foot", min(confidence, 1.0)
        elif hand_count > 0 and fullbody_count == 0:
            # Hands detected but NOT full body (finger tapping or hand movement)
            confidence = hand_count / total_frames
            return "hand", min(confidence, 1.0)
        elif foot_count > 0:
            # Some foot detection
            confidence = foot_count / (total_frames * 2)
            return "foot", min(confidence, 1.0)
        else:
            return "unknown", 0.0


# Example usage
if __name__ == "__main__":
    detector = MovementBasedROI(fps=30)

    # Example video path
    video_path = "test_video.mp4"

    # Detect ROI
    result = detector.detect(video_path)

    print("\n=== ROI Detection Result ===")
    print(f"ROI: {result.roi}")
    print(f"Motion Pattern: {result.motion_pattern}")
    print(f"Motion Area: {result.motion_area_ratio:.1%}")
    print(f"Body Part: {result.body_part}")
    print(f"Confidence: {result.confidence:.1%}")
