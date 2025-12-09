"""
Metrics Calculator for Kinematic Analysis
Calculates clinical metrics from MediaPipe skeleton data

Research References:
- Finger Tapping: PMC10674854 (2023), VisionMD (2025)
- Gait: PMC11597901 (2024), PMC7293393 (2020)
- MDS-UPDRS: Movement Disorder Society Guidelines
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass

# ===== INTERPOLATION CONSTANTS =====
VISIBILITY_THRESHOLD = 0.5  # Minimum visibility score for valid landmark
MAX_INTERPOLATION_GAP = 10  # Maximum consecutive frames to interpolate

# ===== NUMERICAL SAFETY CONSTANTS =====
EPSILON = 1e-8  # Small value to prevent division by zero
PERCENTAGE_CLIP_MIN = -500.0  # Minimum percentage value (clip outliers)
PERCENTAGE_CLIP_MAX = 500.0   # Maximum percentage value (clip outliers)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with epsilon to prevent overflow"""
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator


def clip_percentage(value: float) -> float:
    """Clip percentage values to reasonable range"""
    return np.clip(value, PERCENTAGE_CLIP_MIN, PERCENTAGE_CLIP_MAX)


def interpolate_missing_landmarks(
    landmark_frames: List[Dict],
    required_ids: List[int],
    visibility_threshold: float = VISIBILITY_THRESHOLD,
    max_gap: int = MAX_INTERPOLATION_GAP
) -> List[Dict]:
    """Interpolate missing or low-visibility landmarks using linear interpolation."""
    if len(landmark_frames) < 3:
        return landmark_frames

    interpolated_frames = []
    for frame in landmark_frames:
        new_frame = {
            'frame_number': frame.get('frame_number', 0),
            'timestamp': frame.get('timestamp', 0),
            'keypoints': [kp.copy() for kp in frame.get('keypoints', frame.get('landmarks', []))],
        }
        if frame.get('world_keypoints') or frame.get('world_landmarks'):
            world_kp = frame.get('world_keypoints', frame.get('world_landmarks', []))
            new_frame['world_keypoints'] = [kp.copy() for kp in world_kp] if world_kp else []
        interpolated_frames.append(new_frame)

    for landmark_id in required_ids:
        _interpolate_landmark_trajectory(interpolated_frames, landmark_id, 'keypoints', visibility_threshold, max_gap)
        _interpolate_landmark_trajectory(interpolated_frames, landmark_id, 'world_keypoints', visibility_threshold, max_gap)

    return interpolated_frames


def _interpolate_landmark_trajectory(frames: List[Dict], landmark_id: int, keypoint_key: str, visibility_threshold: float, max_gap: int) -> None:
    """Interpolate a single landmark's trajectory in-place."""
    n_frames = len(frames)
    valid_mask = []
    for frame in frames:
        keypoints = frame.get(keypoint_key, [])
        kp_dict = {kp['id']: kp for kp in keypoints}
        if landmark_id in kp_dict:
            valid_mask.append(kp_dict[landmark_id].get('visibility', 1.0) >= visibility_threshold)
        else:
            valid_mask.append(False)

    i = 0
    while i < n_frames:
        if not valid_mask[i]:
            gap_start = i
            while i < n_frames and not valid_mask[i]:
                i += 1
            gap_end = i
            gap_length = gap_end - gap_start

            if gap_start > 0 and gap_end < n_frames and gap_length <= max_gap:
                start_kp = {kp['id']: kp for kp in frames[gap_start - 1].get(keypoint_key, [])}
                end_kp = {kp['id']: kp for kp in frames[gap_end].get(keypoint_key, [])}

                if landmark_id in start_kp and landmark_id in end_kp:
                    start_lm, end_lm = start_kp[landmark_id], end_kp[landmark_id]
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_length + 1)
                        interpolated_kp = {
                            'id': landmark_id,
                            'x': start_lm['x'] + t * (end_lm['x'] - start_lm['x']),
                            'y': start_lm['y'] + t * (end_lm['y'] - start_lm['y']),
                            'z': start_lm.get('z', 0) + t * (end_lm.get('z', 0) - start_lm.get('z', 0)),
                            'visibility': min(start_lm.get('visibility', 1.0), end_lm.get('visibility', 1.0)),
                            'interpolated': True
                        }
                        frame_kps = frames[j].get(keypoint_key, [])
                        existing_idx = next((idx for idx, kp in enumerate(frame_kps) if kp['id'] == landmark_id), None)
                        if existing_idx is not None:
                            frame_kps[existing_idx] = interpolated_kp
                        else:
                            frame_kps.append(interpolated_kp)
                        frames[j][keypoint_key] = frame_kps
        else:
            i += 1



@dataclass
class FingerTappingMetrics:
    """
    Finger tapping analysis metrics (MDS-UPDRS aligned)

    Research-based features from:
    - PMC10674854: "Clinically Informed Automated Assessment"
    - VisionMD (2025): Normalization methods
    - MDS-UPDRS: Clinical scoring criteria
    """
    # Basic metrics
    tapping_speed: float          # Hz - taps per second
    total_taps: int               # Total number of taps detected
    duration: float               # seconds

    # Amplitude metrics (normalized by index finger length)
    amplitude_mean: float         # Mean opening amplitude (dimensionless)
    amplitude_std: float          # Amplitude variability

    # Velocity metrics (NEW - MDS-UPDRS critical)
    opening_velocity_mean: float  # Mean velocity during finger opening (units/s)
    closing_velocity_mean: float  # Mean velocity during finger closing (units/s)
    peak_velocity_mean: float     # Mean peak velocity per tap
    velocity_decrement: float     # % decrease in velocity over time

    # Decrement analysis (MDS-UPDRS scoring aligned)
    amplitude_decrement: float    # % decrease from first to last tap
    decrement_pattern: str        # "none", "late", "mid", "early" (MDS-UPDRS 0-3)
    first_half_amplitude: float   # Mean amplitude of first 5 taps
    second_half_amplitude: float  # Mean amplitude of last 5 taps

    # Rhythm and hesitation (MDS-UPDRS)
    rhythm_variability: float     # CV of inter-tap intervals (%)
    halt_count: int               # Number of halts (interval > 2x mean)
    hesitation_count: int         # Number of amplitude drops > 20%
    freeze_episodes: int          # Complete stops (interval > 3x mean)

    # Fatigue analysis
    fatigue_rate: float           # % amplitude decrease per tap

    # === TIME-SERIES FEATURES (NEW) ===
    # Temporal segment analysis (thirds)
    velocity_first_third: float   # Mean velocity in first 1/3 of taps
    velocity_mid_third: float     # Mean velocity in middle 1/3
    velocity_last_third: float    # Mean velocity in last 1/3
    amplitude_first_third: float  # Mean amplitude in first 1/3
    amplitude_mid_third: float    # Mean amplitude in middle 1/3
    amplitude_last_third: float   # Mean amplitude in last 1/3

    # Trend analysis (linear regression slopes)
    velocity_slope: float         # Rate of velocity change per tap (negative = declining)
    amplitude_slope: float        # Rate of amplitude change per tap
    rhythm_slope: float           # Rate of rhythm variability change

    # Variability progression
    variability_first_half: float # CV of inter-tap intervals in first half
    variability_second_half: float# CV in second half
    variability_change: float     # % change in variability (positive = worsening)


@dataclass
class GaitMetrics:
    """
    Gait analysis metrics (Clinical standards aligned)

    Research-based features from:
    - PMC11597901: 3D Markerless Motion Capture Validation
    - PMC7293393: Freezing of Gait review
    - PMC6373367: Festination phenotypes
    """
    # Basic metrics
    walking_speed: float          # m/s (estimated)
    cadence: float                # steps/min
    step_count: int               # Total steps detected
    duration: float               # seconds

    # Stride metrics
    stride_length: float          # m (estimated)
    stride_variability: float     # CV of stride time (%)

    # Gait phases (NEW - clinical standard)
    swing_time_mean: float        # Mean swing phase duration (s)
    stance_time_mean: float       # Mean stance phase duration (s)
    swing_stance_ratio: float     # Swing time / Stance time
    double_support_time: float    # Time both feet on ground (s)
    double_support_percent: float # % of gait cycle in double support

    # Asymmetry metrics (NEW - PD specific)
    step_length_asymmetry: float  # % difference between left/right steps
    swing_time_asymmetry: float   # % difference in swing time L/R
    arm_swing_asymmetry: float    # % asymmetry in arm swing amplitude

    # Left/Right breakdown
    left_step_count: int
    right_step_count: int
    left_stride_length: float
    right_stride_length: float

    # Advanced PD features
    festination_index: float      # Degree of step shortening with speed increase
    gait_regularity: float        # Autocorrelation-based regularity (0-1)

    # NEW: Additional PD-specific metrics (World Landmarks based)
    arm_swing_amplitude_left: float   # Left arm swing amplitude in meters
    arm_swing_amplitude_right: float  # Right arm swing amplitude in meters
    arm_swing_amplitude_mean: float   # Mean arm swing amplitude in meters
    step_height_left: float           # Left foot lift height in meters
    step_height_right: float          # Right foot lift height in meters
    step_height_mean: float           # Mean step height in meters

    # NEW: Joint angle features (World Landmarks based, degrees)
    # ROM = Range of Motion (max - min during gait cycle)
    trunk_flexion_mean: float         # Mean trunk forward flexion angle (degrees)
    trunk_flexion_rom: float          # Trunk flexion ROM (degrees)
    hip_flexion_rom_left: float       # Left hip flexion-extension ROM (degrees)
    hip_flexion_rom_right: float      # Right hip flexion-extension ROM (degrees)
    hip_flexion_rom_mean: float       # Mean hip ROM (degrees)
    knee_flexion_rom_left: float      # Left knee flexion ROM (degrees)
    knee_flexion_rom_right: float     # Right knee flexion ROM (degrees)
    knee_flexion_rom_mean: float      # Mean knee ROM (degrees)
    ankle_dorsiflexion_rom_left: float  # Left ankle dorsi/plantarflexion ROM (degrees)
    ankle_dorsiflexion_rom_right: float # Right ankle ROM (degrees)
    ankle_dorsiflexion_rom_mean: float  # Mean ankle ROM (degrees)

    # === TIME-SERIES FEATURES (NEW) ===
    # Step progression (trend over gait cycles)
    step_length_first_half: float     # Mean step length in first half
    step_length_second_half: float    # Mean step length in second half
    step_length_trend: float          # % change (positive = improvement, negative = festination)
    
    # Cadence progression
    cadence_first_half: float         # Cadence in first half (steps/min)
    cadence_second_half: float        # Cadence in second half
    cadence_trend: float              # % change
    
    # Arm swing progression
    arm_swing_first_half: float       # Arm swing amplitude in first half (m)
    arm_swing_second_half: float      # Arm swing in second half
    arm_swing_trend: float            # % change
    
    # Variability progression
    stride_variability_first_half: float  # CV of stride time in first half
    stride_variability_second_half: float # CV in second half
    variability_trend: float              # % change (positive = worsening)
    
    # Step height progression
    step_height_first_half: float     # Mean step height in first half (m)
    step_height_second_half: float    # Mean step height in second half
    step_height_trend: float          # % change


class MetricsCalculator:
    """Calculate clinical metrics from skeleton data using 3D coordinates"""

    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: Video frame rate
        """
        self.fps = fps

    def calculate_finger_tapping_metrics(self, landmark_frames: List[Dict]) -> FingerTappingMetrics:
        """
        Calculate finger tapping metrics from hand landmarks (3D)

        Uses 3D Euclidean distance for camera-angle independence.

        Hand landmarks (MediaPipe):
        - 0: Wrist
        - 4: Thumb tip
        - 5: Index MCP, 6: Index PIP, 7: Index DIP, 8: Index tip
        """
        if not landmark_frames:
            raise ValueError("No landmark data provided")

        # Extract 3D trajectories
        thumb_index_distances = []
        index_finger_lengths = []
        timestamps = []
        wrist_positions = []

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            kp_dict = {kp['id']: kp for kp in keypoints}

            # Required landmarks for finger tapping
            required = [0, 4, 5, 6, 7, 8]
            if all(k in kp_dict for k in required):
                # 3D positions
                wrist = np.array([kp_dict[0]['x'], kp_dict[0]['y'], kp_dict[0]['z']])
                thumb_tip = np.array([kp_dict[4]['x'], kp_dict[4]['y'], kp_dict[4]['z']])
                index_tip = np.array([kp_dict[8]['x'], kp_dict[8]['y'], kp_dict[8]['z']])
                index_mcp = np.array([kp_dict[5]['x'], kp_dict[5]['y'], kp_dict[5]['z']])
                index_pip = np.array([kp_dict[6]['x'], kp_dict[6]['y'], kp_dict[6]['z']])
                index_dip = np.array([kp_dict[7]['x'], kp_dict[7]['y'], kp_dict[7]['z']])

                # 3D Euclidean distance between thumb and index
                distance = np.linalg.norm(thumb_tip - index_tip)
                thumb_index_distances.append(distance)

                # Index finger length for normalization (3D)
                seg1 = np.linalg.norm(index_pip - index_mcp)
                seg2 = np.linalg.norm(index_dip - index_pip)
                seg3 = np.linalg.norm(index_tip - index_dip)
                index_finger_lengths.append(seg1 + seg2 + seg3)

                wrist_positions.append(wrist)

                ts = frame.get('timestamp', frame.get('frame_number', frame.get('frame', 0)) / self.fps)
                timestamps.append(ts)

        if len(thumb_index_distances) < 10:
            raise ValueError("Insufficient landmark data for analysis")

        distances = np.array(thumb_index_distances)
        finger_lengths = np.array(index_finger_lengths)
        timestamps = np.array(timestamps)

        # Normalization factor (VisionMD: max index finger length)
        norm_factor = np.max(finger_lengths) if len(finger_lengths) > 0 else 1.0
        distances_normalized = distances / norm_factor

        duration = timestamps[-1] - timestamps[0]
        dt = 1.0 / self.fps

        # Smooth signal for better peak detection
        if len(distances_normalized) > 11:
            distances_smooth = savgol_filter(distances_normalized, 11, 3)
        else:
            distances_smooth = distances_normalized

        # Detect taps (peaks = maximum opening)
        peaks, peak_props = find_peaks(
            distances_smooth,
            height=np.mean(distances_smooth),
            distance=int(self.fps * 0.08),  # Min 80ms between taps
            prominence=0.05 * np.ptp(distances_smooth)
        )

        # Detect valleys (minimum = finger closed)
        valleys, _ = find_peaks(
            -distances_smooth,
            distance=int(self.fps * 0.08),
            prominence=0.05 * np.ptp(distances_smooth)
        )

        total_taps = len(peaks)
        tapping_speed = total_taps / duration if duration > 0 else 0

        # ===== AMPLITUDE METRICS =====
        if len(peaks) > 0:
            amplitudes = distances_normalized[peaks]
            amplitude_mean = float(np.mean(amplitudes))
            amplitude_std = float(np.std(amplitudes))
        else:
            amplitudes = np.array([])
            amplitude_mean = 0.0
            amplitude_std = 0.0

        # ===== VELOCITY METRICS (NEW) =====
        # Calculate velocity as derivative of distance
        velocity = np.diff(distances_normalized) / dt

        opening_velocities = []
        closing_velocities = []
        peak_velocities = []

        # For each tap cycle, find opening and closing phases
        for i, peak_idx in enumerate(peaks):
            # Find preceding valley (start of opening)
            prev_valleys = valleys[valleys < peak_idx]
            if len(prev_valleys) > 0:
                valley_before = prev_valleys[-1]
                # Opening phase velocity
                if valley_before < len(velocity) and peak_idx <= len(velocity):
                    opening_vel = velocity[valley_before:peak_idx]
                    if len(opening_vel) > 0:
                        opening_velocities.append(np.max(opening_vel))

            # Find following valley (end of closing)
            next_valleys = valleys[valleys > peak_idx]
            if len(next_valleys) > 0:
                valley_after = next_valleys[0]
                # Closing phase velocity (negative values)
                if peak_idx < len(velocity) and valley_after <= len(velocity):
                    closing_vel = velocity[peak_idx:valley_after]
                    if len(closing_vel) > 0:
                        closing_velocities.append(abs(np.min(closing_vel)))

            # Peak velocity for this tap
            window_start = max(0, peak_idx - int(self.fps * 0.1))
            window_end = min(len(velocity), peak_idx + int(self.fps * 0.1))
            if window_start < window_end:
                peak_velocities.append(np.max(np.abs(velocity[window_start:window_end])))

        opening_velocity_mean = float(np.mean(opening_velocities)) if opening_velocities else 0.0
        closing_velocity_mean = float(np.mean(closing_velocities)) if closing_velocities else 0.0
        peak_velocity_mean = float(np.mean(peak_velocities)) if peak_velocities else 0.0

        # Velocity decrement (first half vs second half) - with safe division
        if len(peak_velocities) >= 4:
            half = len(peak_velocities) // 2
            first_half_vel = np.mean(peak_velocities[:half])
            second_half_vel = np.mean(peak_velocities[half:])
            velocity_decrement = clip_percentage(
                safe_divide((first_half_vel - second_half_vel) * 100, first_half_vel, 0.0)
            )
        else:
            velocity_decrement = 0.0

        # ===== AMPLITUDE DECREMENT (MDS-UPDRS aligned) =====
        if len(amplitudes) >= 4:
            half = len(amplitudes) // 2
            first_half_amp = float(np.mean(amplitudes[:half]))
            second_half_amp = float(np.mean(amplitudes[half:]))

            # Safe division with clipping
            amplitude_decrement = clip_percentage(
                safe_divide((first_half_amp - second_half_amp) * 100, first_half_amp, 0.0)
            )

            # Determine decrement pattern (MDS-UPDRS criteria)
            # Score 1: decrement near end (last 30%)
            # Score 2: decrement midway (40-70%)
            # Score 3: decrement after first tap
            if len(amplitudes) >= 6:
                third = len(amplitudes) // 3
                first_third = np.mean(amplitudes[:third])
                mid_third = np.mean(amplitudes[third:2*third])
                last_third = np.mean(amplitudes[2*third:])

                if first_third > 0:
                    early_drop = (first_third - mid_third) / first_third
                    late_drop = (mid_third - last_third) / mid_third if mid_third > 0 else 0

                    if early_drop > 0.15:  # >15% drop after first third
                        decrement_pattern = "early"  # MDS-UPDRS 3
                    elif late_drop > 0.15 and early_drop < 0.1:
                        decrement_pattern = "late"   # MDS-UPDRS 1
                    elif early_drop > 0.08 or late_drop > 0.08:
                        decrement_pattern = "mid"    # MDS-UPDRS 2
                    else:
                        decrement_pattern = "none"   # MDS-UPDRS 0
                else:
                    decrement_pattern = "none"
            else:
                decrement_pattern = "none"
        else:
            first_half_amp = amplitude_mean
            second_half_amp = amplitude_mean
            amplitude_decrement = 0.0
            decrement_pattern = "none"

        # ===== RHYTHM AND HESITATION =====
        if len(peaks) > 2:
            inter_tap_intervals = np.diff(timestamps[peaks])
            mean_interval = np.mean(inter_tap_intervals)
            std_interval = np.std(inter_tap_intervals)

            rhythm_variability = (std_interval / mean_interval * 100) if mean_interval > 0 else 0.0

            # Halt: interval > 1.5x mean (MDS-UPDRS "interruption/hesitation")
            # Lowered from 2x to catch subtle hesitations seen in Score 3 cases
            halt_count = int(np.sum(inter_tap_intervals > mean_interval * 1.5))

            # Freeze: interval > 2.5x mean (longer pause/complete stop)
            freeze_episodes = int(np.sum(inter_tap_intervals > mean_interval * 2.5))
        else:
            rhythm_variability = 0.0
            halt_count = 0
            freeze_episodes = 0

        # Hesitation: amplitude drop > 20% from expected
        hesitation_count = 0
        if len(amplitudes) > 2:
            for i in range(1, len(amplitudes) - 1):
                expected = (amplitudes[i-1] + amplitudes[i+1]) / 2
                if amplitudes[i] < expected * 0.8:
                    hesitation_count += 1

        # ===== FATIGUE RATE =====
        if len(amplitudes) >= 4:
            x = np.arange(len(amplitudes))
            slope = np.polyfit(x, amplitudes, 1)[0]
            fatigue_rate = abs(clip_percentage(safe_divide(slope * 100, amplitude_mean, 0.0)))
        else:
            fatigue_rate = 0.0

        # ===== TIME-SERIES FEATURES (NEW) =====
        # Temporal segment analysis (divide into thirds)
        if len(peak_velocities) >= 6:
            third = len(peak_velocities) // 3
            velocity_first_third = float(np.mean(peak_velocities[:third]))
            velocity_mid_third = float(np.mean(peak_velocities[third:2*third]))
            velocity_last_third = float(np.mean(peak_velocities[2*third:]))
        else:
            velocity_first_third = peak_velocity_mean
            velocity_mid_third = peak_velocity_mean
            velocity_last_third = peak_velocity_mean

        if len(amplitudes) >= 6:
            third = len(amplitudes) // 3
            amplitude_first_third = float(np.mean(amplitudes[:third]))
            amplitude_mid_third = float(np.mean(amplitudes[third:2*third]))
            amplitude_last_third = float(np.mean(amplitudes[2*third:]))
        else:
            amplitude_first_third = amplitude_mean
            amplitude_mid_third = amplitude_mean
            amplitude_last_third = amplitude_mean

        # Trend analysis (linear regression slopes - normalized by mean) - with safe division
        if len(peak_velocities) >= 4:
            x = np.arange(len(peak_velocities))
            vel_slope_raw = np.polyfit(x, peak_velocities, 1)[0]
            velocity_slope = clip_percentage(safe_divide(vel_slope_raw * 100, peak_velocity_mean, 0.0))
        else:
            velocity_slope = 0.0

        if len(amplitudes) >= 4:
            x = np.arange(len(amplitudes))
            amp_slope_raw = np.polyfit(x, amplitudes, 1)[0]
            amplitude_slope = clip_percentage(safe_divide(amp_slope_raw * 100, amplitude_mean, 0.0))
        else:
            amplitude_slope = 0.0

        # Rhythm slope (change in interval variability over time)
        if len(peaks) > 6:
            inter_tap_intervals = np.diff(timestamps[peaks])
            half = len(inter_tap_intervals) // 2
            
            # Calculate rolling CV in windows
            window_size = max(3, len(inter_tap_intervals) // 4)
            cvs = []
            for i in range(0, len(inter_tap_intervals) - window_size + 1, max(1, window_size // 2)):
                window = inter_tap_intervals[i:i+window_size]
                if len(window) >= 2 and np.mean(window) > 0:
                    cvs.append(np.std(window) / np.mean(window) * 100)
            
            if len(cvs) >= 3:
                x = np.arange(len(cvs))
                rhythm_slope = np.polyfit(x, cvs, 1)[0]
            else:
                rhythm_slope = 0.0
                
            # Variability in first vs second half
            first_half_intervals = inter_tap_intervals[:half]
            second_half_intervals = inter_tap_intervals[half:]
            
            if len(first_half_intervals) >= 2 and np.mean(first_half_intervals) > 0:
                variability_first_half = np.std(first_half_intervals) / np.mean(first_half_intervals) * 100
            else:
                variability_first_half = 0.0
                
            if len(second_half_intervals) >= 2 and np.mean(second_half_intervals) > 0:
                variability_second_half = np.std(second_half_intervals) / np.mean(second_half_intervals) * 100
            else:
                variability_second_half = 0.0
                
            # Variability change (positive = worsening) - with safe division
            variability_change = clip_percentage(
                safe_divide(
                    (variability_second_half - variability_first_half) * 100,
                    variability_first_half,
                    default=0.0
                )
            )
        else:
            rhythm_slope = 0.0
            variability_first_half = rhythm_variability
            variability_second_half = rhythm_variability
            variability_change = 0.0

        return FingerTappingMetrics(
            tapping_speed=float(tapping_speed),
            total_taps=int(total_taps),
            duration=float(duration),
            amplitude_mean=amplitude_mean,
            amplitude_std=amplitude_std,
            opening_velocity_mean=opening_velocity_mean,
            closing_velocity_mean=closing_velocity_mean,
            peak_velocity_mean=peak_velocity_mean,
            velocity_decrement=float(velocity_decrement),
            amplitude_decrement=float(amplitude_decrement),
            decrement_pattern=decrement_pattern,
            first_half_amplitude=first_half_amp,
            second_half_amplitude=second_half_amp,
            rhythm_variability=float(rhythm_variability),
            halt_count=int(halt_count),
            hesitation_count=int(hesitation_count),
            freeze_episodes=int(freeze_episodes),
            fatigue_rate=float(max(0, fatigue_rate)),
            # Time-series features
            velocity_first_third=float(velocity_first_third),
            velocity_mid_third=float(velocity_mid_third),
            velocity_last_third=float(velocity_last_third),
            amplitude_first_third=float(amplitude_first_third),
            amplitude_mid_third=float(amplitude_mid_third),
            amplitude_last_third=float(amplitude_last_third),
            velocity_slope=float(velocity_slope),
            amplitude_slope=float(amplitude_slope),
            rhythm_slope=float(rhythm_slope),
            variability_first_half=float(variability_first_half),
            variability_second_half=float(variability_second_half),
            variability_change=float(variability_change)
        )

    def calculate_gait_metrics(self, landmark_frames: List[Dict]) -> GaitMetrics:
        """
        Calculate gait metrics from pose landmarks (3D)

        Uses World Landmarks (real meters) when available for accurate measurements.
        Falls back to normalized landmarks with heuristic scaling if world_landmarks not available.

        Pose landmarks (MediaPipe):
        - 23/24: Left/Right hip
        - 25/26: Left/Right knee
        - 27/28: Left/Right ankle
        - 31/32: Left/Right foot index (toe)
        - 15/16: Left/Right wrist
        """
        if not landmark_frames:
            raise ValueError("No landmark data provided")

        # ===== INTERPOLATION: Fill missing/occluded landmarks =====
        # 11/12: shoulders, 23/24: hips, 25/26: knees, 27/28: ankles, 15/16: wrists, 31/32: toes
        GAIT_REQUIRED_IDS = [11, 12, 23, 24, 25, 26, 27, 28, 15, 16, 31, 32]
        landmark_frames = interpolate_missing_landmarks(
            landmark_frames,
            required_ids=GAIT_REQUIRED_IDS,
            visibility_threshold=VISIBILITY_THRESHOLD,
            max_gap=MAX_INTERPOLATION_GAP
        )

        # Check if world_landmarks are available
        has_world_landmarks = any(
            frame.get('world_keypoints') or frame.get('world_landmarks')
            for frame in landmark_frames
        )

        # Extract 3D trajectories
        left_ankle = []
        right_ankle = []
        left_hip = []
        right_hip = []
        left_wrist = []
        right_wrist = []
        timestamps = []

        # For world landmarks (meters)
        left_ankle_world = []
        right_ankle_world = []
        left_hip_world = []
        right_hip_world = []
        left_wrist_world = []
        right_wrist_world = []
        # NEW: Additional landmarks for joint angle calculation
        left_shoulder_world = []
        right_shoulder_world = []
        left_knee_world = []
        right_knee_world = []
        left_toe_world = []
        right_toe_world = []

        for frame in landmark_frames:
            # Normalized landmarks (for phase detection)
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            kp_dict = {kp['id']: kp for kp in keypoints}

            required = [23, 24, 27, 28]
            if all(k in kp_dict for k in required):
                left_ankle.append([kp_dict[27]['x'], kp_dict[27]['y'], kp_dict[27]['z']])
                right_ankle.append([kp_dict[28]['x'], kp_dict[28]['y'], kp_dict[28]['z']])
                left_hip.append([kp_dict[23]['x'], kp_dict[23]['y'], kp_dict[23]['z']])
                right_hip.append([kp_dict[24]['x'], kp_dict[24]['y'], kp_dict[24]['z']])

                if 15 in kp_dict and 16 in kp_dict:
                    left_wrist.append([kp_dict[15]['x'], kp_dict[15]['y'], kp_dict[15]['z']])
                    right_wrist.append([kp_dict[16]['x'], kp_dict[16]['y'], kp_dict[16]['z']])

                # World landmarks (real meters) - for distance calculations
                world_kp = frame.get('world_keypoints', frame.get('world_landmarks'))
                if world_kp:
                    world_dict = {kp['id']: kp for kp in world_kp}
                    if all(k in world_dict for k in required):
                        left_ankle_world.append([world_dict[27]['x'], world_dict[27]['y'], world_dict[27]['z']])
                        right_ankle_world.append([world_dict[28]['x'], world_dict[28]['y'], world_dict[28]['z']])
                        left_hip_world.append([world_dict[23]['x'], world_dict[23]['y'], world_dict[23]['z']])
                        right_hip_world.append([world_dict[24]['x'], world_dict[24]['y'], world_dict[24]['z']])
                        # Wrists for arm swing (World Landmarks)
                        if 15 in world_dict and 16 in world_dict:
                            left_wrist_world.append([world_dict[15]['x'], world_dict[15]['y'], world_dict[15]['z']])
                            right_wrist_world.append([world_dict[16]['x'], world_dict[16]['y'], world_dict[16]['z']])
                        # NEW: Shoulders for trunk angle (11: left, 12: right)
                        if 11 in world_dict and 12 in world_dict:
                            left_shoulder_world.append([world_dict[11]['x'], world_dict[11]['y'], world_dict[11]['z']])
                            right_shoulder_world.append([world_dict[12]['x'], world_dict[12]['y'], world_dict[12]['z']])
                        # NEW: Knees for knee angle (25: left, 26: right)
                        if 25 in world_dict and 26 in world_dict:
                            left_knee_world.append([world_dict[25]['x'], world_dict[25]['y'], world_dict[25]['z']])
                            right_knee_world.append([world_dict[26]['x'], world_dict[26]['y'], world_dict[26]['z']])
                        # NEW: Toes for ankle angle (31: left, 32: right)
                        if 31 in world_dict and 32 in world_dict:
                            left_toe_world.append([world_dict[31]['x'], world_dict[31]['y'], world_dict[31]['z']])
                            right_toe_world.append([world_dict[32]['x'], world_dict[32]['y'], world_dict[32]['z']])

                ts = frame.get('timestamp', frame.get('frame_number', frame.get('frame', 0)) / self.fps)
                timestamps.append(ts)

        if len(left_ankle) < 10:
            raise ValueError("Insufficient gait data for analysis")

        left_ankle = np.array(left_ankle)
        right_ankle = np.array(right_ankle)
        left_hip = np.array(left_hip)
        right_hip = np.array(right_hip)
        timestamps = np.array(timestamps)

        duration = timestamps[-1] - timestamps[0]
        dt = 1.0 / self.fps

        # Hip center (3D)
        hip_center = (left_hip + right_hip) / 2

        # ===== GAIT PHASE DETECTION =====
        # Method: Zeni et al. (2008) - Heel position relative to hip
        # Heel Strike = when heel is at maximum forward position relative to hip (Z-axis)
        # Toe Off = when toe is at maximum backward position relative to hip
        # Reference: PMC2384115, MDPI Sensors 23(14):6489

        # Use World Landmarks for accurate 3D positions (meters)
        use_world_for_gait = len(left_ankle_world) > 10 and len(right_ankle_world) > 10

        if use_world_for_gait:
            # World Landmarks: Z-axis = forward/backward (sagittal plane)
            left_ankle_arr_w = np.array(left_ankle_world)
            right_ankle_arr_w = np.array(right_ankle_world)
            left_hip_arr_w = np.array(left_hip_world[:len(left_ankle_arr_w)])
            right_hip_arr_w = np.array(right_hip_world[:len(right_ankle_arr_w)])
            hip_center_world = (left_hip_arr_w + right_hip_arr_w) / 2

            # Zeni method: heel Z position relative to hip center
            # Positive = foot in front of hip, Negative = foot behind hip
            left_heel_rel_z = left_ankle_arr_w[:, 2] - hip_center_world[:, 2]
            right_heel_rel_z = right_ankle_arr_w[:, 2] - hip_center_world[:, 2]

            # Apply smoothing (Butterworth-like effect with Savgol)
            if len(left_heel_rel_z) > 15:
                left_heel_rel_z = savgol_filter(left_heel_rel_z, 15, 3)
                right_heel_rel_z = savgol_filter(right_heel_rel_z, 15, 3)
        else:
            # Fallback: use normalized landmarks
            left_heel_rel_z = left_ankle[:, 2] - hip_center[:, 2]
            right_heel_rel_z = right_ankle[:, 2] - hip_center[:, 2]
            if len(left_heel_rel_z) > 15:
                left_heel_rel_z = savgol_filter(left_heel_rel_z, 15, 3)
                right_heel_rel_z = savgol_filter(right_heel_rel_z, 15, 3)

        # Heel-strikes = local maxima (foot furthest forward)
        # Min 400ms between same-foot heel strikes (normal gait ~500-600ms per stride)
        min_stride_frames = int(self.fps * 0.4)

        left_hs, _ = find_peaks(
            left_heel_rel_z,
            distance=min_stride_frames,
            prominence=0.03 * np.ptp(left_heel_rel_z)  # 3% of total range
        )

        right_hs, _ = find_peaks(
            right_heel_rel_z,
            distance=min_stride_frames,
            prominence=0.03 * np.ptp(right_heel_rel_z)
        )

        # Toe-offs = local minima (foot furthest backward)
        left_to, _ = find_peaks(
            -left_heel_rel_z,
            distance=min_stride_frames,
            prominence=0.03 * np.ptp(left_heel_rel_z)
        )

        right_to, _ = find_peaks(
            -right_heel_rel_z,
            distance=min_stride_frames,
            prominence=0.03 * np.ptp(right_heel_rel_z)
        )

        # For backward compatibility, also calculate hip-ankle distance
        left_hip_ankle_dist = np.sqrt(
            (hip_center[:, 0] - left_ankle[:, 0]) ** 2 +
            (hip_center[:, 2] - left_ankle[:, 2]) ** 2
        )
        right_hip_ankle_dist = np.sqrt(
            (hip_center[:, 0] - right_ankle[:, 0]) ** 2 +
            (hip_center[:, 2] - right_ankle[:, 2]) ** 2
        )

        left_step_count = len(left_hs)
        right_step_count = len(right_hs)
        total_steps = left_step_count + right_step_count

        # Cadence
        cadence = (total_steps / duration) * 60 if duration > 0 else 0

        # ===== SWING AND STANCE TIMES =====
        left_swing_times = []
        left_stance_times = []
        right_swing_times = []
        right_stance_times = []

        # Left foot: swing = TO to HS, stance = HS to next TO
        for i, hs in enumerate(left_hs):
            # Find preceding toe-off
            prev_to = left_to[left_to < hs]
            if len(prev_to) > 0:
                to_idx = prev_to[-1]
                swing_time = (timestamps[hs] - timestamps[to_idx])
                if 0.1 < swing_time < 1.5:  # Valid range
                    left_swing_times.append(swing_time)

            # Find following toe-off
            next_to = left_to[left_to > hs]
            if len(next_to) > 0:
                to_idx = next_to[0]
                stance_time = (timestamps[to_idx] - timestamps[hs])
                if 0.2 < stance_time < 2.0:
                    left_stance_times.append(stance_time)

        # Right foot
        for i, hs in enumerate(right_hs):
            prev_to = right_to[right_to < hs]
            if len(prev_to) > 0:
                to_idx = prev_to[-1]
                swing_time = (timestamps[hs] - timestamps[to_idx])
                if 0.1 < swing_time < 1.5:
                    right_swing_times.append(swing_time)

            next_to = right_to[right_to > hs]
            if len(next_to) > 0:
                to_idx = next_to[0]
                stance_time = (timestamps[to_idx] - timestamps[hs])
                if 0.2 < stance_time < 2.0:
                    right_stance_times.append(stance_time)

        all_swing_times = left_swing_times + right_swing_times
        all_stance_times = left_stance_times + right_stance_times

        swing_time_mean = float(np.mean(all_swing_times)) if all_swing_times else 0.0
        stance_time_mean = float(np.mean(all_stance_times)) if all_stance_times else 0.0
        swing_stance_ratio = swing_time_mean / stance_time_mean if stance_time_mean > 0 else 0.0

        # Double support time (both feet on ground)
        # Approximate: overlap between stance phases
        gait_cycle_time = swing_time_mean + stance_time_mean if (swing_time_mean + stance_time_mean) > 0 else 1.0
        # In normal gait, double support is ~20% of cycle
        # Estimate from stance time: double_support ≈ 2 * (stance - swing)
        double_support_time = max(0, (stance_time_mean - swing_time_mean))
        double_support_percent = (double_support_time / gait_cycle_time * 100) if gait_cycle_time > 0 else 0

        # ===== STRIDE LENGTH AND SPEED =====
        # IMPORTANT: World Landmarks are hip-centered (local coordinates relative to hip)
        # so they cannot be used for stride length calculation (position doesn't change globally)
        # Always use normalized landmarks with hip-width heuristic for stride length

        # Use normalized landmarks for stride length calculation
        left_ankle_arr = left_ankle
        right_ankle_arr = right_ankle
        hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
        avg_hip_width_norm = np.mean(hip_widths[hip_widths > 0.01])
        REAL_HIP_WIDTH = 0.30  # meters (average adult hip width)
        scale_factor = REAL_HIP_WIDTH / avg_hip_width_norm if avg_hip_width_norm > 0 else 2.0
        scale_factor = np.clip(scale_factor, 1.0, 15.0)

        # Calculate per-step lengths
        left_step_lengths = []
        right_step_lengths = []

        # Left steps: distance between consecutive left heel-strikes
        for i in range(1, len(left_hs)):
            if left_hs[i] < len(left_ankle_arr) and left_hs[i-1] < len(left_ankle_arr):
                step_dist = np.linalg.norm(left_ankle_arr[left_hs[i]] - left_ankle_arr[left_hs[i-1]])
                step_length = step_dist * scale_factor
                if 0.1 < step_length < 2.5:  # Valid range (meters)
                    left_step_lengths.append(step_length)

        for i in range(1, len(right_hs)):
            if right_hs[i] < len(right_ankle_arr) and right_hs[i-1] < len(right_ankle_arr):
                step_dist = np.linalg.norm(right_ankle_arr[right_hs[i]] - right_ankle_arr[right_hs[i-1]])
                step_length = step_dist * scale_factor
                if 0.1 < step_length < 2.5:
                    right_step_lengths.append(step_length)

        left_stride_length = float(np.mean(left_step_lengths)) if left_step_lengths else 0.0
        right_stride_length = float(np.mean(right_step_lengths)) if right_step_lengths else 0.0

        all_step_lengths = left_step_lengths + right_step_lengths
        stride_length = float(np.mean(all_step_lengths)) if all_step_lengths else 0.6

        # Walking speed = cadence * stride_length
        walking_speed = (cadence / 60) * stride_length if cadence > 0 else 0.0

        # Validate walking speed (typical range: 0.5 - 2.0 m/s for adults)
        if walking_speed < 0.2 or walking_speed > 3.0:
            # Fallback to typical values
            walking_speed = (cadence / 60) * 0.6
            stride_length = 0.6

        # ===== ASYMMETRY METRICS ===== (with safe division)
        # Step length asymmetry
        if left_stride_length > 0 and right_stride_length > 0:
            avg_stride = (left_stride_length + right_stride_length) / 2
            step_length_asymmetry = clip_percentage(
                safe_divide(abs(left_stride_length - right_stride_length) * 100, avg_stride, 0.0)
            )
        else:
            step_length_asymmetry = 0.0

        # Swing time asymmetry
        left_swing_mean = np.mean(left_swing_times) if left_swing_times else 0
        right_swing_mean = np.mean(right_swing_times) if right_swing_times else 0
        if left_swing_mean > 0 and right_swing_mean > 0:
            avg_swing = (left_swing_mean + right_swing_mean) / 2
            swing_time_asymmetry = clip_percentage(
                safe_divide(abs(left_swing_mean - right_swing_mean) * 100, avg_swing, 0.0)
            )
        else:
            swing_time_asymmetry = 0.0

        # ===== ARM SWING (World Landmarks - meters) =====
        # Use World Landmarks to avoid camera perspective error
        # Calculate per-gait-cycle to avoid measuring global movement
        use_world_wrist = len(left_wrist_world) > 10 and len(right_wrist_world) > 10

        if use_world_wrist:
            # Use World Landmarks (real meters)
            left_wrist_arr = np.array(left_wrist_world)
            right_wrist_arr = np.array(right_wrist_world)
            left_hip_arr = np.array(left_hip_world[:len(left_wrist_arr)])
            right_hip_arr = np.array(right_hip_world[:len(right_wrist_arr)])
            hip_center_world = (left_hip_arr + right_hip_arr) / 2

            # Arm swing amplitude = peak-to-peak range of wrist position relative to hip
            # Using Z-axis (forward/backward in World Landmarks) for sagittal plane movement
            left_wrist_rel_z = left_wrist_arr[:, 2] - hip_center_world[:, 2]
            right_wrist_rel_z = right_wrist_arr[:, 2] - hip_center_world[:, 2]

            # Calculate per-gait-cycle amplitude (not global peak-to-peak)
            # Use heel strikes to define gait cycles
            left_cycle_amplitudes = []
            right_cycle_amplitudes = []

            # Combine all heel strikes and sort
            all_hs = sorted(list(left_hs) + list(right_hs))

            for i in range(1, len(all_hs)):
                start_idx = all_hs[i-1]
                end_idx = all_hs[i]

                # Only process if indices are valid
                if end_idx <= len(left_wrist_rel_z) and start_idx < end_idx:
                    # Left arm amplitude in this cycle
                    left_cycle_z = left_wrist_rel_z[start_idx:end_idx]
                    if len(left_cycle_z) > 5:
                        left_amp = np.ptp(left_cycle_z)
                        if 0.01 < left_amp < 1.0:  # Valid range: 1cm to 50cm
                            left_cycle_amplitudes.append(left_amp)

                    # Right arm amplitude in this cycle
                    right_cycle_z = right_wrist_rel_z[start_idx:end_idx]
                    if len(right_cycle_z) > 5:
                        right_amp = np.ptp(right_cycle_z)
                        if 0.01 < right_amp < 1.0:
                            right_cycle_amplitudes.append(right_amp)

            # Average amplitude across all cycles
            left_arm_amplitude = float(np.mean(left_cycle_amplitudes)) if left_cycle_amplitudes else 0.0
            right_arm_amplitude = float(np.mean(right_cycle_amplitudes)) if right_cycle_amplitudes else 0.0
            arm_swing_amplitude_mean = (left_arm_amplitude + right_arm_amplitude) / 2

            # Asymmetry calculation (with safe division)
            if left_arm_amplitude > 0 and right_arm_amplitude > 0:
                more = max(left_arm_amplitude, right_arm_amplitude)
                less = min(left_arm_amplitude, right_arm_amplitude)
                arm_swing_asymmetry = clip_percentage(safe_divide((more - less) * 100, more, 0.0))
            else:
                arm_swing_asymmetry = 0.0
        elif len(left_wrist) > 0 and len(right_wrist) > 0:
            # Fallback: Use normalized landmarks (less accurate)
            left_wrist_norm = np.array(left_wrist)
            right_wrist_norm = np.array(right_wrist)

            # X-axis displacement relative to hip center
            left_wrist_rel = left_wrist_norm[:, 0] - hip_center[:len(left_wrist_norm), 0]
            right_wrist_rel = right_wrist_norm[:, 0] - hip_center[:len(right_wrist_norm), 0]

            left_arm_range_norm = np.ptp(left_wrist_rel)
            right_arm_range_norm = np.ptp(right_wrist_rel)

            # Estimate amplitude in meters using hip-width scaling
            hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
            avg_hip_width = np.mean(hip_widths[hip_widths > 0.01])
            REAL_HIP_WIDTH = 0.30  # meters
            scale = REAL_HIP_WIDTH / avg_hip_width if avg_hip_width > 0 else 2.0

            left_arm_amplitude = left_arm_range_norm * scale
            right_arm_amplitude = right_arm_range_norm * scale
            arm_swing_amplitude_mean = (left_arm_amplitude + right_arm_amplitude) / 2

            if left_arm_range_norm > 0 and right_arm_range_norm > 0:
                more = max(left_arm_range_norm, right_arm_range_norm)
                less = min(left_arm_range_norm, right_arm_range_norm)
                arm_swing_asymmetry = clip_percentage(safe_divide((more - less) * 100, more, 0.0))
            else:
                arm_swing_asymmetry = 0.0
        else:
            arm_swing_asymmetry = 0.0
            left_arm_amplitude = 0.0
            right_arm_amplitude = 0.0
            arm_swing_amplitude_mean = 0.0

        # ===== STEP HEIGHT (World Landmarks - meters) =====
        # Step height = maximum vertical lift of ankle during swing phase
        # Uses Y-axis in World Landmarks (vertical, with Y increasing upward in hip-centered coords)
        if len(left_ankle_world) > 10:
            left_ankle_arr_world = np.array(left_ankle_world)
            right_ankle_arr_world = np.array(right_ankle_world)

            # For each foot, find peaks in Y position (highest point during swing)
            # World Landmarks: Y is vertical (positive = up relative to hip center)
            # Note: In World Landmarks, hip center is origin, so ankle Y will be negative (below hip)

            # Calculate step heights per swing phase
            left_step_heights = []
            right_step_heights = []

            # Left foot step heights: find Y peak between each toe-off and heel-strike
            for i, hs in enumerate(left_hs):
                prev_to = left_to[left_to < hs]
                if len(prev_to) > 0:
                    to_idx = prev_to[-1]
                    if to_idx < len(left_ankle_arr_world) and hs < len(left_ankle_arr_world):
                        # Y values during swing (to_idx to hs)
                        swing_y = left_ankle_arr_world[to_idx:hs, 1]
                        if len(swing_y) > 2:
                            # Step height = peak Y - baseline Y (min of endpoints)
                            baseline_y = min(left_ankle_arr_world[to_idx, 1],
                                           left_ankle_arr_world[hs-1, 1] if hs > 0 else left_ankle_arr_world[to_idx, 1])
                            peak_y = np.max(swing_y)
                            step_height = peak_y - baseline_y
                            if 0.01 < step_height < 0.50:  # Valid range: 1cm to 50cm
                                left_step_heights.append(step_height)

            # Right foot step heights
            for i, hs in enumerate(right_hs):
                prev_to = right_to[right_to < hs]
                if len(prev_to) > 0:
                    to_idx = prev_to[-1]
                    if to_idx < len(right_ankle_arr_world) and hs < len(right_ankle_arr_world):
                        swing_y = right_ankle_arr_world[to_idx:hs, 1]
                        if len(swing_y) > 2:
                            baseline_y = min(right_ankle_arr_world[to_idx, 1],
                                           right_ankle_arr_world[hs-1, 1] if hs > 0 else right_ankle_arr_world[to_idx, 1])
                            peak_y = np.max(swing_y)
                            step_height = peak_y - baseline_y
                            if 0.01 < step_height < 0.50:
                                right_step_heights.append(step_height)

            step_height_left = float(np.mean(left_step_heights)) if left_step_heights else 0.0
            step_height_right = float(np.mean(right_step_heights)) if right_step_heights else 0.0
            step_height_mean = (step_height_left + step_height_right) / 2 if (step_height_left + step_height_right) > 0 else 0.0
        else:
            step_height_left = 0.0
            step_height_right = 0.0
            step_height_mean = 0.0

        # ===== STRIDE VARIABILITY =====
        all_hs = np.sort(np.concatenate([left_hs, right_hs]))
        if len(all_hs) >= 4:
            step_intervals = np.diff(timestamps[all_hs])
            mean_interval = np.mean(step_intervals)
            # Filter outliers
            valid = step_intervals[(step_intervals > mean_interval * 0.5) &
                                   (step_intervals < mean_interval * 2.0)]
            if len(valid) >= 2:
                stride_variability = (np.std(valid) / np.mean(valid)) * 100
            else:
                stride_variability = (np.std(step_intervals) / mean_interval) * 100
        else:
            stride_variability = 0.0

        # ===== FESTINATION INDEX =====
        # Festination = steps getting shorter while cadence increases
        if len(all_step_lengths) >= 6:
            half = len(all_step_lengths) // 2
            first_half_steps = all_step_lengths[:half]
            second_half_steps = all_step_lengths[half:]

            first_half_mean = np.mean(first_half_steps)
            second_half_mean = np.mean(second_half_steps)

            # Check if steps are getting shorter
            if first_half_mean > 0:
                step_decrement = (first_half_mean - second_half_mean) / first_half_mean
                # Festination: positive value means steps getting shorter
                festination_index = max(0, step_decrement * 100)
            else:
                festination_index = 0.0
        else:
            festination_index = 0.0

        # ===== GAIT REGULARITY (Autocorrelation) =====
        if len(left_hip_ankle_dist) >= 30:
            # Normalize the signal
            signal = left_hip_ankle_dist - np.mean(left_hip_ankle_dist)
            # Autocorrelation at gait cycle lag
            expected_cycle_frames = int(self.fps * 60 / cadence) if cadence > 0 else int(self.fps)
            expected_cycle_frames = min(expected_cycle_frames, len(signal) // 2)

            if expected_cycle_frames > 5:
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # Normalize

                # Find peak near expected cycle
                search_range = slice(max(0, expected_cycle_frames - 10),
                                    min(len(autocorr), expected_cycle_frames + 10))
                if len(autocorr[search_range]) > 0:
                    gait_regularity = float(np.max(autocorr[search_range]))
                else:
                    gait_regularity = 0.5
            else:
                gait_regularity = 0.5
        else:
            gait_regularity = 0.5

        # ===== JOINT ANGLES (World Landmarks - degrees) =====
        # Helper function to calculate angle between three points
        def calculate_angle_3d(p1, p2, p3):
            """Calculate angle at p2 between vectors p2->p1 and p2->p3 in degrees"""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))

        # Initialize angle values
        trunk_flexion_mean = 0.0
        trunk_flexion_rom = 0.0
        hip_flexion_rom_left = 0.0
        hip_flexion_rom_right = 0.0
        hip_flexion_rom_mean = 0.0
        knee_flexion_rom_left = 0.0
        knee_flexion_rom_right = 0.0
        knee_flexion_rom_mean = 0.0
        ankle_dorsiflexion_rom_left = 0.0
        ankle_dorsiflexion_rom_right = 0.0
        ankle_dorsiflexion_rom_mean = 0.0

        # Check if we have enough data for angle calculations
        min_frames_for_angles = 10
        has_shoulder_data = len(left_shoulder_world) >= min_frames_for_angles
        has_knee_data = len(left_knee_world) >= min_frames_for_angles
        has_toe_data = len(left_toe_world) >= min_frames_for_angles

        if has_shoulder_data and len(left_hip_world) >= min_frames_for_angles:
            # Convert to numpy arrays
            left_shoulder_arr = np.array(left_shoulder_world[:len(left_hip_world)])
            right_shoulder_arr = np.array(right_shoulder_world[:len(left_hip_world)])
            left_hip_arr_w = np.array(left_hip_world)
            right_hip_arr_w = np.array(right_hip_world)

            # Trunk center (midpoint between shoulders)
            trunk_center = (left_shoulder_arr + right_shoulder_arr) / 2
            hip_center_arr = (left_hip_arr_w + right_hip_arr_w) / 2

            # Trunk flexion: angle from vertical
            # Vertical reference: point directly above hip center (Y-axis in World Landmarks)
            trunk_angles = []
            for i in range(len(trunk_center)):
                # Vector from hip to shoulder
                trunk_vec = trunk_center[i] - hip_center_arr[i]
                # Vertical reference (Y-axis up)
                vertical = np.array([0, 1, 0])
                # Angle from vertical (0 = upright, 90 = horizontal)
                cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                trunk_angles.append(angle)

            trunk_angles = np.array(trunk_angles)
            trunk_flexion_mean = float(np.mean(trunk_angles))
            trunk_flexion_rom = float(np.ptp(trunk_angles))

        if has_knee_data and len(left_hip_world) >= min_frames_for_angles and len(left_ankle_world) >= min_frames_for_angles:
            left_hip_arr_w = np.array(left_hip_world)
            right_hip_arr_w = np.array(right_hip_world)
            left_knee_arr = np.array(left_knee_world[:len(left_hip_arr_w)])
            right_knee_arr = np.array(right_knee_world[:len(left_hip_arr_w)])
            left_ankle_arr_w = np.array(left_ankle_world[:len(left_hip_arr_w)])
            right_ankle_arr_w = np.array(right_ankle_world[:len(left_hip_arr_w)])

            # Hip flexion angle: angle at hip between trunk (shoulder) and knee
            if has_shoulder_data:
                left_shoulder_arr = np.array(left_shoulder_world[:len(left_hip_arr_w)])
                right_shoulder_arr = np.array(right_shoulder_world[:len(left_hip_arr_w)])

                left_hip_angles = []
                right_hip_angles = []
                for i in range(len(left_hip_arr_w)):
                    # Left hip angle: shoulder -> hip -> knee
                    left_angle = calculate_angle_3d(left_shoulder_arr[i], left_hip_arr_w[i], left_knee_arr[i])
                    left_hip_angles.append(180 - left_angle)  # Convert to flexion angle
                    # Right hip angle
                    right_angle = calculate_angle_3d(right_shoulder_arr[i], right_hip_arr_w[i], right_knee_arr[i])
                    right_hip_angles.append(180 - right_angle)

                hip_flexion_rom_left = float(np.ptp(left_hip_angles))
                hip_flexion_rom_right = float(np.ptp(right_hip_angles))
                hip_flexion_rom_mean = (hip_flexion_rom_left + hip_flexion_rom_right) / 2

            # Knee flexion angle: angle at knee between hip and ankle
            left_knee_angles = []
            right_knee_angles = []
            for i in range(len(left_knee_arr)):
                # Left knee angle: hip -> knee -> ankle
                left_angle = calculate_angle_3d(left_hip_arr_w[i], left_knee_arr[i], left_ankle_arr_w[i])
                left_knee_angles.append(180 - left_angle)  # Convert to flexion angle
                # Right knee angle
                right_angle = calculate_angle_3d(right_hip_arr_w[i], right_knee_arr[i], right_ankle_arr_w[i])
                right_knee_angles.append(180 - right_angle)

            knee_flexion_rom_left = float(np.ptp(left_knee_angles))
            knee_flexion_rom_right = float(np.ptp(right_knee_angles))
            knee_flexion_rom_mean = (knee_flexion_rom_left + knee_flexion_rom_right) / 2

        if has_toe_data and has_knee_data and len(left_ankle_world) >= min_frames_for_angles:
            left_knee_arr = np.array(left_knee_world[:len(left_ankle_world)])
            right_knee_arr = np.array(right_knee_world[:len(left_ankle_world)])
            left_ankle_arr_w = np.array(left_ankle_world)
            right_ankle_arr_w = np.array(right_ankle_world)
            left_toe_arr = np.array(left_toe_world[:len(left_ankle_world)])
            right_toe_arr = np.array(right_toe_world[:len(left_ankle_world)])

            # Ankle dorsiflexion angle: angle at ankle between knee and toe
            left_ankle_angles = []
            right_ankle_angles = []
            for i in range(len(left_ankle_arr_w)):
                # Left ankle angle: knee -> ankle -> toe
                left_angle = calculate_angle_3d(left_knee_arr[i], left_ankle_arr_w[i], left_toe_arr[i])
                left_ankle_angles.append(left_angle - 90)  # Relative to neutral (90 degrees)
                # Right ankle angle
                right_angle = calculate_angle_3d(right_knee_arr[i], right_ankle_arr_w[i], right_toe_arr[i])
                right_ankle_angles.append(right_angle - 90)

            ankle_dorsiflexion_rom_left = float(np.ptp(left_ankle_angles))
            ankle_dorsiflexion_rom_right = float(np.ptp(right_ankle_angles))
            ankle_dorsiflexion_rom_mean = (ankle_dorsiflexion_rom_left + ankle_dorsiflexion_rom_right) / 2

        # ===== TIME-SERIES FEATURES (NEW) ===== (with safe division)
        # Step length progression
        if len(all_step_lengths) >= 4:
            half = len(all_step_lengths) // 2
            step_length_first_half = float(np.mean(all_step_lengths[:half]))
            step_length_second_half = float(np.mean(all_step_lengths[half:]))
            step_length_trend = clip_percentage(
                safe_divide((step_length_second_half - step_length_first_half) * 100, step_length_first_half, 0.0)
            )
        else:
            step_length_first_half = stride_length
            step_length_second_half = stride_length
            step_length_trend = 0.0

        # Cadence progression (using step intervals)
        all_hs_sorted = np.sort(np.concatenate([left_hs, right_hs]))
        if len(all_hs_sorted) >= 6:
            step_intervals = np.diff(timestamps[all_hs_sorted])
            valid_intervals = step_intervals[(step_intervals > 0.2) & (step_intervals < 2.0)]
            if len(valid_intervals) >= 4:
                half = len(valid_intervals) // 2
                first_cadence = 60.0 / np.mean(valid_intervals[:half]) if np.mean(valid_intervals[:half]) > 0 else 0
                second_cadence = 60.0 / np.mean(valid_intervals[half:]) if np.mean(valid_intervals[half:]) > 0 else 0
                cadence_first_half = float(first_cadence)
                cadence_second_half = float(second_cadence)
                cadence_trend = clip_percentage(
                    safe_divide((cadence_second_half - cadence_first_half) * 100, cadence_first_half, 0.0)
                )
            else:
                cadence_first_half = cadence
                cadence_second_half = cadence
                cadence_trend = 0.0
        else:
            cadence_first_half = cadence
            cadence_second_half = cadence
            cadence_trend = 0.0

        # Arm swing progression
        if use_world_wrist and len(left_cycle_amplitudes) >= 4:
            all_arm_amps = [(l + r) / 2 for l, r in zip(left_cycle_amplitudes[:len(right_cycle_amplitudes)], right_cycle_amplitudes)]
            if len(all_arm_amps) >= 4:
                half = len(all_arm_amps) // 2
                arm_swing_first_half = float(np.mean(all_arm_amps[:half]))
                arm_swing_second_half = float(np.mean(all_arm_amps[half:]))
                arm_swing_trend = clip_percentage(
                    safe_divide((arm_swing_second_half - arm_swing_first_half) * 100, arm_swing_first_half, 0.0)
                )
            else:
                arm_swing_first_half = arm_swing_amplitude_mean
                arm_swing_second_half = arm_swing_amplitude_mean
                arm_swing_trend = 0.0
        else:
            arm_swing_first_half = arm_swing_amplitude_mean
            arm_swing_second_half = arm_swing_amplitude_mean
            arm_swing_trend = 0.0

        # Stride variability progression
        if len(all_hs_sorted) >= 8:
            step_intervals = np.diff(timestamps[all_hs_sorted])
            valid_intervals = step_intervals[(step_intervals > 0.2) & (step_intervals < 2.0)]
            if len(valid_intervals) >= 6:
                half = len(valid_intervals) // 2
                first_std = np.std(valid_intervals[:half])
                first_mean = np.mean(valid_intervals[:half])
                second_std = np.std(valid_intervals[half:])
                second_mean = np.mean(valid_intervals[half:])
                
                stride_variability_first_half = clip_percentage(safe_divide(first_std * 100, first_mean, 0.0))
                stride_variability_second_half = clip_percentage(safe_divide(second_std * 100, second_mean, 0.0))

                variability_trend = clip_percentage(
                    safe_divide((stride_variability_second_half - stride_variability_first_half) * 100,
                               stride_variability_first_half, 0.0)
                )
            else:
                stride_variability_first_half = stride_variability
                stride_variability_second_half = stride_variability
                variability_trend = 0.0
        else:
            stride_variability_first_half = stride_variability
            stride_variability_second_half = stride_variability
            variability_trend = 0.0

        # Step height progression
        all_step_heights = left_step_heights + right_step_heights if 'left_step_heights' in dir() else []
        if len(all_step_heights) >= 4:
            half = len(all_step_heights) // 2
            step_height_first_half = float(np.mean(all_step_heights[:half]))
            step_height_second_half = float(np.mean(all_step_heights[half:]))
            step_height_trend = clip_percentage(
                safe_divide((step_height_second_half - step_height_first_half) * 100, step_height_first_half, 0.0)
            )
        else:
            step_height_first_half = step_height_mean
            step_height_second_half = step_height_mean
            step_height_trend = 0.0

        return GaitMetrics(
            walking_speed=float(walking_speed),
            cadence=float(cadence),
            step_count=int(total_steps),
            duration=float(duration),
            stride_length=float(stride_length),
            stride_variability=float(stride_variability),
            swing_time_mean=float(swing_time_mean),
            stance_time_mean=float(stance_time_mean),
            swing_stance_ratio=float(swing_stance_ratio),
            double_support_time=float(double_support_time),
            double_support_percent=float(double_support_percent),
            step_length_asymmetry=float(step_length_asymmetry),
            swing_time_asymmetry=float(swing_time_asymmetry),
            arm_swing_asymmetry=float(arm_swing_asymmetry),
            left_step_count=int(left_step_count),
            right_step_count=int(right_step_count),
            left_stride_length=float(left_stride_length),
            right_stride_length=float(right_stride_length),
            festination_index=float(festination_index),
            gait_regularity=float(np.clip(gait_regularity, 0, 1)),
            # NEW: World Landmarks based metrics
            arm_swing_amplitude_left=float(left_arm_amplitude),
            arm_swing_amplitude_right=float(right_arm_amplitude),
            arm_swing_amplitude_mean=float(arm_swing_amplitude_mean),
            step_height_left=float(step_height_left),
            step_height_right=float(step_height_right),
            step_height_mean=float(step_height_mean),
            # NEW: Joint angle features (degrees)
            trunk_flexion_mean=float(trunk_flexion_mean),
            trunk_flexion_rom=float(trunk_flexion_rom),
            hip_flexion_rom_left=float(hip_flexion_rom_left),
            hip_flexion_rom_right=float(hip_flexion_rom_right),
            hip_flexion_rom_mean=float(hip_flexion_rom_mean),
            knee_flexion_rom_left=float(knee_flexion_rom_left),
            knee_flexion_rom_right=float(knee_flexion_rom_right),
            knee_flexion_rom_mean=float(knee_flexion_rom_mean),
            ankle_dorsiflexion_rom_left=float(ankle_dorsiflexion_rom_left),
            ankle_dorsiflexion_rom_right=float(ankle_dorsiflexion_rom_right),
            ankle_dorsiflexion_rom_mean=float(ankle_dorsiflexion_rom_mean),
            # Time-series features
            step_length_first_half=float(step_length_first_half),
            step_length_second_half=float(step_length_second_half),
            step_length_trend=float(step_length_trend),
            cadence_first_half=float(cadence_first_half),
            cadence_second_half=float(cadence_second_half),
            cadence_trend=float(cadence_trend),
            arm_swing_first_half=float(arm_swing_first_half),
            arm_swing_second_half=float(arm_swing_second_half),
            arm_swing_trend=float(arm_swing_trend),
            stride_variability_first_half=float(stride_variability_first_half),
            stride_variability_second_half=float(stride_variability_second_half),
            variability_trend=float(variability_trend),
            step_height_first_half=float(step_height_first_half),
            step_height_second_half=float(step_height_second_half),
            step_height_trend=float(step_height_trend)
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

    with open(skeleton_path, 'r') as f:
        landmark_frames = json.load(f)

    print(f"\nLoaded {len(landmark_frames)} frames from {skeleton_path}")

    calculator = MetricsCalculator(fps=30.0)

    if mode == "finger":
        metrics = calculator.calculate_finger_tapping_metrics(landmark_frames)
        print("\n=== Finger Tapping Metrics ===")
        print(f"Tapping Speed: {metrics.tapping_speed:.2f} Hz")
        print(f"Total Taps: {metrics.total_taps}")
        print(f"Amplitude (mean): {metrics.amplitude_mean:.3f}")
        print(f"Opening Velocity: {metrics.opening_velocity_mean:.3f} /s")
        print(f"Closing Velocity: {metrics.closing_velocity_mean:.3f} /s")
        print(f"Velocity Decrement: {metrics.velocity_decrement:.1f}%")
        print(f"Amplitude Decrement: {metrics.amplitude_decrement:.1f}%")
        print(f"Decrement Pattern: {metrics.decrement_pattern}")
        print(f"Halt Count: {metrics.halt_count}")
        print(f"Hesitation Count: {metrics.hesitation_count}")
        print(f"Freeze Episodes: {metrics.freeze_episodes}")

    elif mode == "gait":
        metrics = calculator.calculate_gait_metrics(landmark_frames)
        print("\n=== Gait Metrics ===")
        print(f"Walking Speed: {metrics.walking_speed:.2f} m/s")
        print(f"Cadence: {metrics.cadence:.1f} steps/min")
        print(f"Stride Length: {metrics.stride_length:.2f} m")
        print(f"Swing Time: {metrics.swing_time_mean:.3f} s")
        print(f"Stance Time: {metrics.stance_time_mean:.3f} s")
        print(f"Swing/Stance Ratio: {metrics.swing_stance_ratio:.2f}")
        print(f"Double Support: {metrics.double_support_percent:.1f}%")
        print(f"Step Asymmetry: {metrics.step_length_asymmetry:.1f}%")
        print(f"Arm Swing Asymmetry: {metrics.arm_swing_asymmetry:.1f}%")
        print(f"Festination Index: {metrics.festination_index:.1f}%")
        print(f"Gait Regularity: {metrics.gait_regularity:.2f}")
