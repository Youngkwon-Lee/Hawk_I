"""
Metrics Calculator for Kinematic Analysis
Calculates clinical metrics from MediaPipe skeleton data
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from dataclasses import dataclass


@dataclass
class FingerTappingMetrics:
    """Finger tapping analysis metrics"""
    tapping_speed: float  # Hz
    amplitude_mean: float  # pixels
    amplitude_std: float  # pixels
    rhythm_variability: float  # %
    fatigue_rate: float  # %
    hesitation_count: int
    total_taps: int
    duration: float  # seconds


@dataclass
class GaitMetrics:
    """Gait analysis metrics"""
    walking_speed: float  # m/s (estimated)
    stride_length: float  # m (estimated)
    cadence: float  # steps/min
    stride_variability: float  # %
    arm_swing_asymmetry: float  # %
    step_count: int
    duration: float  # seconds


class MetricsCalculator:
    """Calculate clinical metrics from skeleton data"""

    def __init__(self, fps: float = 30.0, video_resolution: Tuple[int, int] = (854, 480)):
        """
        Args:
            fps: Video frame rate
            video_resolution: (width, height) in pixels
        """
        self.fps = fps
        self.video_width, self.video_height = video_resolution

    def calculate_finger_tapping_metrics(self, landmark_frames: List[Dict]) -> FingerTappingMetrics:
        """
        Calculate finger tapping metrics from hand landmarks (3D)

        Research-based method from:
        "Clinically Informed Automated Assessment of Finger Tapping Videos in Parkinson's Disease"
        (PMC10674854, 2023)

        Hand landmarks (21 points):
        - 4: Thumb tip (K4)
        - 8: Index finger tip (K8)

        Key improvement: Thumb-Index distance instead of single finger tracking
        """
        if not landmark_frames:
            raise ValueError("No landmark data provided")

        # Extract thumb-index distance and calculate normalization factor (VisionMD method)
        # Normalization factor: INDEX finger length (sum of 3 segments)
        thumb_index_distances = []
        index_finger_lengths = []
        timestamps = []

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', [])
            kp_dict = {kp['id']: kp for kp in keypoints}

            # Need thumb tip (4), index tip (8), and index segments (5,6,7,8) for normalization
            if 4 in kp_dict and 8 in kp_dict and 5 in kp_dict and 6 in kp_dict and 7 in kp_dict:
                thumb_tip = np.array([kp_dict[4]['x'], kp_dict[4]['y'], kp_dict[4]['z']])
                index_tip = np.array([kp_dict[8]['x'], kp_dict[8]['y'], kp_dict[8]['z']])

                # Calculate 3D Euclidean distance between thumb and index
                distance = np.linalg.norm(thumb_tip - index_tip)
                thumb_index_distances.append(distance)

                # Calculate INDEX normalization factor (sum of 3 index finger segments)
                # MediaPipe landmarks: 5=MCP, 6=PIP, 7=DIP, 8=TIP
                index_mcp = np.array([kp_dict[5]['x'], kp_dict[5]['y'], kp_dict[5]['z']])
                index_pip = np.array([kp_dict[6]['x'], kp_dict[6]['y'], kp_dict[6]['z']])
                index_dip = np.array([kp_dict[7]['x'], kp_dict[7]['y'], kp_dict[7]['z']])

                seg1 = np.linalg.norm(index_pip - index_mcp)  # MCP-PIP
                seg2 = np.linalg.norm(index_dip - index_pip)  # PIP-DIP
                seg3 = np.linalg.norm(index_tip - index_dip)  # DIP-TIP

                index_length = seg1 + seg2 + seg3
                index_finger_lengths.append(index_length)

                timestamps.append(frame.get('timestamp', frame['frame'] / self.fps))

        if len(thumb_index_distances) < 10:
            raise ValueError("Insufficient landmark data for analysis")

        thumb_index_distances = np.array(thumb_index_distances)
        index_finger_lengths = np.array(index_finger_lengths)
        timestamps = np.array(timestamps)

        # Calculate normalization factor (VisionMD method: max INDEX length)
        normalization_factor = np.max(index_finger_lengths) if len(index_finger_lengths) > 0 else 1.0

        # Detect taps (peaks in thumb-index distance)
        # Peaks = maximum opening (finger extended)
        peaks, properties = find_peaks(
            thumb_index_distances,
            height=np.mean(thumb_index_distances),
            distance=int(self.fps * 0.1)  # Minimum 0.1s between taps
        )

        total_taps = len(peaks)
        duration = timestamps[-1] - timestamps[0]

        # Tapping speed (Hz)
        tapping_speed = total_taps / duration if duration > 0 else 0

        # Amplitude (peak thumb-index distances, normalized by index finger length)
        # VisionMD method: divide by INDEX finger length for hand-size independence
        if len(peaks) > 0:
            # Extract amplitude at each peak
            amplitudes_raw = thumb_index_distances[peaks]

            # Normalize by index finger length (dimensionless)
            amplitudes_normalized = amplitudes_raw / normalization_factor

            amplitude_mean = np.mean(amplitudes_normalized)
            amplitude_std = np.std(amplitudes_normalized)
        else:
            amplitude_mean = 0
            amplitude_std = 0
            amplitudes_normalized = np.array([])

        # Rhythm variability (coefficient of variation of inter-tap intervals)
        if len(peaks) > 2:
            inter_tap_intervals = np.diff(timestamps[peaks])
            rhythm_variability = (np.std(inter_tap_intervals) / np.mean(inter_tap_intervals)) * 100
        else:
            rhythm_variability = 0

        # Fatigue rate (amplitude decrement using piecewise linear regression)
        # Research: Use slope of amplitude trend to detect decrement
        if len(peaks) >= 4 and len(amplitudes_normalized) >= 4:
            # Simple linear regression for amplitude trend
            x = np.arange(len(amplitudes_normalized))
            y = amplitudes_normalized

            # Calculate slope (negative = decreasing amplitude)
            slope = np.polyfit(x, y, 1)[0]

            # Normalize by initial amplitude to get percentage
            if np.mean(y) > 0:
                fatigue_rate = abs(slope / np.mean(y)) * 100 * len(y)
            else:
                fatigue_rate = 0
        else:
            fatigue_rate = 0

        # Hesitation count (research-based: 20% threshold)
        # Reference: PMC10674854 - "threshold α = 0.2"
        hesitation_count = 0
        if len(peaks) > 2 and len(amplitudes_normalized) > 2:
            # Calculate expected amplitude based on adjacent values
            for i in range(1, len(amplitudes_normalized) - 1):
                # Average of adjacent amplitudes
                expected_amp = (amplitudes_normalized[i-1] + amplitudes_normalized[i+1]) / 2

                # Hesitation: actual amplitude < 80% of expected (20% deviation)
                if amplitudes_normalized[i] < expected_amp * 0.8:
                    hesitation_count += 1

        return FingerTappingMetrics(
            tapping_speed=tapping_speed,
            amplitude_mean=amplitude_mean,
            amplitude_std=amplitude_std,
            rhythm_variability=rhythm_variability,
            fatigue_rate=max(0, fatigue_rate),  # Clamp to 0
            hesitation_count=int(hesitation_count),
            total_taps=total_taps,
            duration=duration
        )

    def calculate_gait_metrics(self, landmark_frames: List[Dict]) -> GaitMetrics:
        """
        Calculate gait metrics from pose landmarks (3D)

        Pose landmarks (33 points):
        - 23: Left hip
        - 24: Right hip
        - 27: Left ankle
        - 28: Right ankle
        - 15: Left wrist
        - 16: Right wrist
        """
        if not landmark_frames:
            raise ValueError("No landmark data provided")

        # Extract 3D trajectories
        left_ankle_y = []
        right_ankle_y = []
        hip_center_3d = []  # (x, y, z)
        left_wrist_x = []
        right_wrist_x = []
        timestamps = []

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', [])
            kp_dict = {kp['id']: kp for kp in keypoints}

            if 27 in kp_dict and 28 in kp_dict and 23 in kp_dict and 24 in kp_dict:
                left_ankle_y.append(kp_dict[27]['y'])
                right_ankle_y.append(kp_dict[28]['y'])

                # Hip center 3D position
                hip_x = (kp_dict[23]['x'] + kp_dict[24]['x']) / 2
                hip_y = (kp_dict[23]['y'] + kp_dict[24]['y']) / 2
                hip_z = (kp_dict[23]['z'] + kp_dict[24]['z']) / 2
                hip_center_3d.append([hip_x, hip_y, hip_z])

                # Arm swing
                if 15 in kp_dict and 16 in kp_dict:
                    left_wrist_x.append(kp_dict[15]['x'])
                    right_wrist_x.append(kp_dict[16]['x'])

                timestamps.append(frame.get('timestamp', frame['frame'] / self.fps))

        if len(left_ankle_y) < 10:
            raise ValueError("Insufficient gait data for analysis")

        left_ankle_y = np.array(left_ankle_y)
        right_ankle_y = np.array(right_ankle_y)
        hip_center_3d = np.array(hip_center_3d)  # Shape: (N, 3)
        timestamps = np.array(timestamps)

        duration = timestamps[-1] - timestamps[0]

        # Calculate hip-foot horizontal distance for step detection (research-based method)
        # Paper: "Automated Gait Analysis Based on a Marker-Free Pose Estimation Model"
        # Method: Relative horizontal distance between hip and foot

        left_foot_3d = []
        right_foot_3d = []

        # Re-extract foot positions
        for frame in landmark_frames:
            keypoints = frame.get('keypoints', [])
            kp_dict = {kp['id']: kp for kp in keypoints}

            if 27 in kp_dict and 28 in kp_dict:
                left_foot_3d.append([kp_dict[27]['x'], kp_dict[27]['y'], kp_dict[27]['z']])
                right_foot_3d.append([kp_dict[28]['x'], kp_dict[28]['y'], kp_dict[28]['z']])

        left_foot_3d = np.array(left_foot_3d)
        right_foot_3d = np.array(right_foot_3d)

        # Calculate horizontal distance between hip center and each foot
        left_hip_foot_dist = np.sqrt(
            (hip_center_3d[:, 0] - left_foot_3d[:, 0]) ** 2 +
            (hip_center_3d[:, 2] - left_foot_3d[:, 2]) ** 2  # X-Z plane (horizontal)
        )

        right_hip_foot_dist = np.sqrt(
            (hip_center_3d[:, 0] - right_foot_3d[:, 0]) ** 2 +
            (hip_center_3d[:, 2] - right_foot_3d[:, 2]) ** 2  # X-Z plane (horizontal)
        )

        # Detect heel-strikes (peaks) and toe-offs (valleys)
        # Heel-strike = maximum hip-foot distance
        # Toe-off = minimum hip-foot distance

        # Left foot heel-strikes
        left_peak_threshold = np.max(left_hip_foot_dist) * 0.35  # 35% of max (from paper)
        left_steps, _ = find_peaks(
            left_hip_foot_dist,
            height=left_peak_threshold,
            distance=int(self.fps * 0.8)  # 0.8s minimum between steps (from paper)
        )

        # Right foot heel-strikes
        right_peak_threshold = np.max(right_hip_foot_dist) * 0.46  # 46% of max (from paper)
        right_steps, _ = find_peaks(
            right_hip_foot_dist,
            height=right_peak_threshold,
            distance=int(self.fps * 0.8)  # 0.8s minimum between steps
        )

        total_steps = len(left_steps) + len(right_steps)

        # Cadence (steps per minute)
        cadence = (total_steps / duration) * 60 if duration > 0 else 0

        # Estimate walking speed based on cadence and average stride length
        # Normal walking: stride length ≈ 0.7-0.8m, speed ≈ 1.0-1.4 m/s
        # Use empirical relationship: speed = cadence * stride_length / 60

        # Estimate stride length from video duration and step count
        # Assume person walks continuously in frame
        # For frontal view: use normalized coordinate change
        x_displacement = abs(hip_center_3d[-1, 0] - hip_center_3d[0, 0])
        z_displacement = abs(hip_center_3d[-1, 2] - hip_center_3d[0, 2])

        # Use larger displacement (either lateral or forward)
        max_displacement = max(x_displacement, z_displacement)

        # Calibration: Typical person height ≈ 1.7m
        # Hip height ≈ 0.53 * body height ≈ 0.9m
        # Normal stride ≈ 0.7-0.8m
        # Scale normalized coordinates to meters
        # Assume normalized coordinate range [0, 1] ≈ 2m in real world
        scale_factor = 2.0
        distance_traveled = max_displacement * scale_factor

        walking_speed = distance_traveled / duration if duration > 0 else 0

        # Stride length (distance per step)
        stride_length = distance_traveled / total_steps if total_steps > 0 else 0

        # Alternative cadence-based speed estimation (more reliable)
        if cadence > 0 and total_steps >= 4:
            # Use typical stride length relationship: stride ≈ 0.43 * height
            # Assume average height = 1.7m → stride ≈ 0.73m
            typical_stride = 0.73
            cadence_based_speed = (cadence * typical_stride) / 60

            # Use cadence-based speed if displacement-based seems unreliable
            if walking_speed < 0.1 or walking_speed > 3.0:
                walking_speed = cadence_based_speed
                stride_length = typical_stride

        # Stride variability
        if len(left_steps) > 2:
            left_stride_intervals = np.diff(timestamps[left_steps])
            stride_variability = (np.std(left_stride_intervals) / np.mean(left_stride_intervals)) * 100
        else:
            stride_variability = 0

        # Arm swing asymmetry (research-based)
        # Reference: "Evaluation of Arm Swing Features and Asymmetry during Gait in
        # Parkinson's Disease Using the Azure Kinect Sensor" (PMC9412494, 2022)
        # Method: Wrist displacement relative to pelvis
        if len(left_wrist_x) > 0 and len(right_wrist_x) > 0:
            left_wrist_x = np.array(left_wrist_x)
            right_wrist_x = np.array(right_wrist_x)

            # Extract pelvis center (hip center already calculated)
            pelvis_x = hip_center_3d[:, 0]

            # Calculate wrist position relative to pelvis
            left_wrist_relative = left_wrist_x - pelvis_x
            right_wrist_relative = right_wrist_x - pelvis_x

            # Anteroposterior (AP) range: horizontal swing amplitude
            left_arm_range = np.ptp(left_wrist_relative)
            right_arm_range = np.ptp(right_wrist_relative)

            # ASA (Absolute Symmetry Angle) formula
            # ASA = abs((45° - arctan(PMORE/PLESS))/90°) × 100
            if left_arm_range > 0 and right_arm_range > 0:
                more = max(left_arm_range, right_arm_range)
                less = min(left_arm_range, right_arm_range)
                asa_radians = abs((np.pi/4 - np.arctan(more/less)) / (np.pi/2))
                arm_swing_asymmetry = asa_radians * 100
            else:
                arm_swing_asymmetry = 0
        else:
            arm_swing_asymmetry = 0

        return GaitMetrics(
            walking_speed=walking_speed,
            stride_length=stride_length,
            cadence=cadence,
            stride_variability=stride_variability,
            arm_swing_asymmetry=arm_swing_asymmetry,
            step_count=total_steps,
            duration=duration
        )


# Example usage
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 3:
        print("Usage: python metrics_calculator.py <skeleton_json> <mode>")
        print("  mode: 'finger' or 'gait'")
        sys.exit(1)

    skeleton_path = sys.argv[1]
    mode = sys.argv[2]

    # Load skeleton data
    with open(skeleton_path, 'r') as f:
        landmark_frames = json.load(f)

    print(f"\nLoaded {len(landmark_frames)} frames from {skeleton_path}")

    # Calculate metrics
    calculator = MetricsCalculator(fps=25.0)  # Adjust based on video

    if mode == "finger":
        metrics = calculator.calculate_finger_tapping_metrics(landmark_frames)
        print("\n=== Finger Tapping Metrics ===")
        print(f"Tapping Speed: {metrics.tapping_speed:.2f} Hz")
        print(f"Amplitude (mean): {metrics.amplitude_mean:.2f} pixels")
        print(f"Amplitude (std): {metrics.amplitude_std:.2f} pixels")
        print(f"Rhythm Variability: {metrics.rhythm_variability:.1f}%")
        print(f"Fatigue Rate: {metrics.fatigue_rate:.1f}%")
        print(f"Hesitation Count: {metrics.hesitation_count}")
        print(f"Total Taps: {metrics.total_taps}")
        print(f"Duration: {metrics.duration:.2f}s")

    elif mode == "gait":
        metrics = calculator.calculate_gait_metrics(landmark_frames)
        print("\n=== Gait Metrics ===")
        print(f"Walking Speed: {metrics.walking_speed:.2f} m/s")
        print(f"Stride Length: {metrics.stride_length:.2f} m")
        print(f"Cadence: {metrics.cadence:.1f} steps/min")
        print(f"Stride Variability: {metrics.stride_variability:.1f}%")
        print(f"Arm Swing Asymmetry: {metrics.arm_swing_asymmetry:.1f}%")
        print(f"Step Count: {metrics.step_count}")
        print(f"Duration: {metrics.duration:.2f}s")
