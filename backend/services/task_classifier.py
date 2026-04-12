"""
Task Classification Module
Auto-detect task type from ROI information
video_type: 'finger_tapping', 'gait', 'turning', 'leg_agility', 'hand_movement'
"""

from typing import Tuple, Dict
from dataclasses import dataclass
from .roi_detector import ROIResult


@dataclass
class TaskClassificationResult:
    """Task Classification Result"""
    task_type: str  # 'finger_tapping', 'gait', 'leg_agility', 'pronation_supination', etc.
    confidence: float
    reasoning: str
    motion_pattern: str
    motion_area_ratio: float
    roi: Tuple[int, int, int, int]


class TaskClassifier:
    """
    Task Classifier using ROI information and body part detection

    IMPROVED Classification Rules (v2):
    1. Hand + Small Motion Area (< 5%) → Finger Tapping
    2. Hand + Larger Motion Area (≥ 5%) → Hand Movement (Pronation-Supination)
    3. Foot + Large Motion (> 10%) → Gait
    4. Foot + Small Motion (< 10%) → Leg Agility
    5. Fullbody / Global Motion → Gait

    Note: Uses motion_area_ratio instead of ROI size for better accuracy
    """

    def __init__(self):
        pass

    def classify(self, roi_result: ROIResult, video_frame_size: Tuple[int, int]) -> TaskClassificationResult:
        """
        Classify task type from ROI detection result

        Args:
            roi_result: ROI detection result from MovementBasedROI
            video_frame_size: (width, height) of video frame

        Returns:
            TaskClassificationResult with task type and confidence
        """
        body_part = roi_result.body_part
        motion_area_ratio = roi_result.motion_area_ratio
        motion_pattern = roi_result.motion_pattern
        roi = roi_result.roi

        # Calculate ROI size ratio (compared to frame)
        frame_width, frame_height = video_frame_size
        frame_area = frame_width * frame_height
        roi_area = roi[2] * roi[3]  # w * h
        roi_size_ratio = roi_area / frame_area

        print(f"\n=== Task Classification ===")
        print(f"Body Part: {body_part}")
        print(f"Motion Pattern: {motion_pattern}")
        print(f"Motion Area Ratio: {motion_area_ratio:.1%}")
        print(f"ROI Size Ratio: {roi_size_ratio:.1%}")

        # SIMPLIFIED Classification logic (v3)
        # Currently only finger_tapping and gait are fully implemented
        # hand_movement and leg_agility are planned for future
        if body_part == "hand":
            # Hand detected → Always Finger Tapping (hand_movement not implemented yet)
            task_type = "finger_tapping"
            confidence = 0.90
            reasoning = (
                f"Hand detected with motion area ({motion_area_ratio:.1%}). "
                f"Classified as finger_tapping."
            )
            print(f"  → Hand detected → Finger Tapping")

        elif body_part == "foot":
            # Foot detected → Always Gait (leg_agility not implemented yet)
            task_type = "gait"
            confidence = 0.90
            reasoning = (
                f"Foot motion detected with area ({motion_area_ratio:.1%}). "
                f"Classified as gait."
            )
            print(f"  → Foot detected → Gait")

        elif body_part == "fullbody" or motion_pattern == "global":
            # Fullbody or global motion → Gait
            task_type = "gait"
            confidence = 0.90
            reasoning = (
                f"Fullbody motion detected (area: {motion_area_ratio:.1%}). "
                f"Classified as gait."
            )
            print(f"  → Fullbody/Global Motion → Gait")

        else:
            # Unknown body part
            task_type = "unknown"
            confidence = 0.0
            reasoning = (
                f"Unable to detect body part clearly in ROI. "
                f"Task classification failed."
            )
            print(f"  → Unknown body part → Unable to classify")

        print(f"Final Task: {task_type}")
        print(f"Confidence: {confidence:.1%}")

        return TaskClassificationResult(
            task_type=task_type,
            confidence=confidence,
            reasoning=reasoning,
            motion_pattern=motion_pattern,
            motion_area_ratio=motion_area_ratio,
            roi=roi
        )

    def classify_from_video(
        self,
        video_path: str,
        roi_detector  # MovementBasedROI instance
    ) -> TaskClassificationResult:
        """
        Convenience method to classify directly from video

        Args:
            video_path: Path to video file
            roi_detector: MovementBasedROI instance

        Returns:
            TaskClassificationResult
        """
        import cv2

        # Get video frame size
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Detect ROI
        roi_result = roi_detector.detect(video_path)

        # Classify task
        return self.classify(roi_result, (frame_width, frame_height))


# Example usage
if __name__ == "__main__":
    from roi_detector import MovementBasedROI

    # Initialize
    roi_detector = MovementBasedROI(fps=30)
    classifier = TaskClassifier()

    # Example video
    video_path = "test_video.mp4"

    # Classify
    result = classifier.classify_from_video(video_path, roi_detector)

    print("\n=== Classification Result ===")
    print(f"Task Type: {result.task_type}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reasoning: {result.reasoning}")
