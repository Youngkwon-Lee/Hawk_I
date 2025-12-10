"""
Visualization Data Generator
Generates data for frontend visualization charts from skeleton landmarks
Supports both Gait and Finger Tapping tasks with dynamic visualization
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from scipy.signal import savgol_filter, find_peaks


class PopulationStatsLoader:
    """
    Load and compute population statistics from PD4T training data.
    These statistics are used to set adaptive thresholds for event detection.
    """

    _instance = None
    _stats_cache = {}

    # Path to feature files (relative to project root)
    FEATURES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "processed", "features"
    )

    @classmethod
    def get_instance(cls):
        """Singleton pattern for caching stats"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_stats()

    def _load_stats(self):
        """Load statistics from training data"""
        try:
            # Load gait features
            gait_path = os.path.join(self.FEATURES_DIR, "gait_train_features_stratified.csv")
            if os.path.exists(gait_path):
                gait_df = pd.read_csv(gait_path)
                self._stats_cache['gait'] = self._compute_stats(gait_df, task_type='gait')
            else:
                self._stats_cache['gait'] = self._default_gait_stats()

            # Load finger tapping features
            ft_path = os.path.join(self.FEATURES_DIR, "finger_tapping_train_features_stratified.csv")
            if os.path.exists(ft_path):
                ft_df = pd.read_csv(ft_path)
                self._stats_cache['finger_tapping'] = self._compute_stats(ft_df, task_type='finger_tapping')
            else:
                self._stats_cache['finger_tapping'] = self._default_finger_stats()

        except Exception as e:
            print(f"Warning: Could not load population stats: {e}")
            self._stats_cache['gait'] = self._default_gait_stats()
            self._stats_cache['finger_tapping'] = self._default_finger_stats()

    def _compute_stats(self, df: pd.DataFrame, task_type: str) -> Dict:
        """Compute percentile-based thresholds from training data"""
        stats = {}

        if task_type == 'gait':
            # Speed statistics (for pause detection)
            if 'walking_speed' in df.columns:
                speeds = df['walking_speed'].dropna()
                stats['speed_median'] = float(speeds.median())
                stats['speed_p25'] = float(speeds.quantile(0.25))
                stats['speed_p10'] = float(speeds.quantile(0.10))  # Slow threshold
                stats['speed_std'] = float(speeds.std())

            # Asymmetry statistics (for asymmetry event detection)
            for col in ['step_length_asymmetry', 'swing_time_asymmetry', 'arm_swing_asymmetry']:
                if col in df.columns:
                    vals = df[col].dropna().abs()
                    stats[f'{col}_median'] = float(vals.median())
                    stats[f'{col}_p75'] = float(vals.quantile(0.75))  # "Abnormal" threshold
                    stats[f'{col}_p90'] = float(vals.quantile(0.90))  # "High" threshold

            # Variability statistics
            if 'stride_variability' in df.columns:
                vals = df['stride_variability'].dropna()
                stats['variability_median'] = float(vals.median())
                stats['variability_p75'] = float(vals.quantile(0.75))

            # Score-based statistics (different thresholds for different severity)
            for score in range(5):
                score_df = df[df['score'] == score]
                if len(score_df) > 5:
                    stats[f'score_{score}_speed_median'] = float(score_df['walking_speed'].median()) if 'walking_speed' in score_df.columns else None
                    stats[f'score_{score}_asymmetry_median'] = float(score_df['step_length_asymmetry'].abs().median()) if 'step_length_asymmetry' in score_df.columns else None

        elif task_type == 'finger_tapping':
            # Amplitude statistics (for fatigue detection)
            if 'amplitude_mean' in df.columns:
                vals = df['amplitude_mean'].dropna()
                stats['amplitude_median'] = float(vals.median())
                stats['amplitude_p25'] = float(vals.quantile(0.25))
                stats['amplitude_std'] = float(vals.std())

            # Fatigue rate statistics
            if 'fatigue_rate' in df.columns:
                vals = df['fatigue_rate'].dropna()
                stats['fatigue_rate_median'] = float(vals.median())
                stats['fatigue_rate_p75'] = float(vals.quantile(0.75))  # Abnormal fatigue threshold
                stats['fatigue_rate_p90'] = float(vals.quantile(0.90))

            # Amplitude decrement (related to fatigue)
            if 'amplitude_decrement' in df.columns:
                vals = df['amplitude_decrement'].dropna()
                stats['amplitude_decrement_median'] = float(vals.median())
                stats['amplitude_decrement_p75'] = float(vals.quantile(0.75))

            # Rhythm variability statistics
            if 'rhythm_variability' in df.columns:
                vals = df['rhythm_variability'].dropna()
                stats['rhythm_variability_median'] = float(vals.median())
                stats['rhythm_variability_p75'] = float(vals.quantile(0.75))  # Rhythm break threshold
                stats['rhythm_variability_p90'] = float(vals.quantile(0.90))

            # Velocity statistics (for pause detection)
            if 'peak_velocity_mean' in df.columns:
                vals = df['peak_velocity_mean'].dropna()
                stats['velocity_median'] = float(vals.median())
                stats['velocity_p25'] = float(vals.quantile(0.25))
                stats['velocity_std'] = float(vals.std())

            # Hesitation/halt statistics (for pause detection)
            for col in ['hesitation_count', 'halt_count', 'freeze_episodes']:
                if col in df.columns:
                    vals = df[col].dropna()
                    stats[f'{col}_mean'] = float(vals.mean())
                    stats[f'{col}_p75'] = float(vals.quantile(0.75))

            # Score-based statistics
            for score in range(5):
                score_df = df[df['score'] == score]
                if len(score_df) > 5:
                    stats[f'score_{score}_fatigue_median'] = float(score_df['fatigue_rate'].median()) if 'fatigue_rate' in score_df.columns else None
                    stats[f'score_{score}_variability_median'] = float(score_df['rhythm_variability'].median()) if 'rhythm_variability' in score_df.columns else None

        return stats

    def _default_gait_stats(self) -> Dict:
        """Default gait statistics if training data not available"""
        return {
            'speed_median': 0.8,
            'speed_p25': 0.6,
            'speed_p10': 0.4,
            'speed_std': 0.15,
            'step_length_asymmetry_median': 5.0,
            'step_length_asymmetry_p75': 10.0,
            'step_length_asymmetry_p90': 20.0,
            'swing_time_asymmetry_median': 8.0,
            'swing_time_asymmetry_p75': 15.0,
            'swing_time_asymmetry_p90': 25.0,
            'variability_median': 25.0,
            'variability_p75': 35.0,
        }

    def _default_finger_stats(self) -> Dict:
        """Default finger tapping statistics if training data not available"""
        return {
            'amplitude_median': 1.2,
            'amplitude_p25': 0.9,
            'amplitude_std': 0.3,
            'fatigue_rate_median': 1.0,
            'fatigue_rate_p75': 2.5,
            'fatigue_rate_p90': 5.0,
            'amplitude_decrement_median': 10.0,
            'amplitude_decrement_p75': 20.0,
            'rhythm_variability_median': 15.0,
            'rhythm_variability_p75': 25.0,
            'rhythm_variability_p90': 40.0,
            'velocity_median': 20.0,
            'velocity_p25': 15.0,
            'velocity_std': 5.0,
        }

    def get_stats(self, task_type: str) -> Dict:
        """Get statistics for a task type"""
        if task_type == 'finger_tapping':
            return self._stats_cache.get('finger_tapping', self._default_finger_stats())
        else:
            return self._stats_cache.get('gait', self._default_gait_stats())

    def get_threshold(self, task_type: str, metric: str, level: str = 'p75') -> float:
        """
        Get a specific threshold value.

        Args:
            task_type: 'gait' or 'finger_tapping'
            metric: e.g., 'fatigue_rate', 'rhythm_variability', 'step_length_asymmetry'
            level: 'median', 'p75', 'p90', etc.

        Returns:
            Threshold value
        """
        stats = self.get_stats(task_type)
        key = f'{metric}_{level}'
        if key in stats:
            return stats[key]
        elif metric in stats:
            return stats[metric]
        else:
            return None


@dataclass
class VisualizationData:
    """Container for all visualization chart data"""
    joint_angles: List[Dict]  # For JointAngleChart / TappingAmplitudeChart
    symmetry: List[Dict]      # For SymmetryChart
    gait_cycles: List[Dict]   # For GaitCycleChart / TappingRhythmChart
    speed_profile: List[Dict] # For SpeedProfileChart


class VisualizationDataGenerator:
    """Generate visualization data from skeleton landmarks"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def generate(self, landmark_frames: List[Dict], gait_analysis: Optional[Dict] = None, task_type: str = "gait") -> Dict:
        """
        Generate all visualization data from landmarks

        Args:
            landmark_frames: List of frame data with landmarks
            gait_analysis: Optional pre-computed gait cycle analysis
            task_type: 'gait' or 'finger_tapping'

        Returns:
            Dict with visualization data for all charts
        """
        if len(landmark_frames) < 10:
            return self._empty_data(task_type)

        if task_type == "finger_tapping":
            # Extract finger/wrist trajectories for finger tapping
            trajectories = self._extract_finger_trajectories(landmark_frames)
            return self._generate_finger_tapping_data(trajectories)
        else:
            # Extract full body trajectories for gait
            trajectories = self._extract_trajectories(landmark_frames)
            return self._generate_gait_data(trajectories, gait_analysis)

    def _empty_data(self, task_type: str = "gait") -> Dict:
        """Return empty data structure"""
        return {
            "task_type": task_type,
            "joint_angles": [],
            "symmetry": [],
            "gait_cycles": [],
            "speed_profile": []
        }

    def _extract_finger_trajectories(self, landmark_frames: List[Dict]) -> Dict:
        """Extract finger/wrist trajectories for finger tapping analysis"""
        trajectories = {
            'timestamps': [],
            'left_wrist': [],
            'right_wrist': [],
            'left_index': [],
            'right_index': [],
            'left_thumb': [],
            'right_thumb': [],
        }

        # MediaPipe Hand landmark IDs (for Pose model, using wrist approximation)
        # Pose model: 15=left_wrist, 16=right_wrist, 17=left_pinky, 18=right_pinky
        # 19=left_index, 20=right_index, 21=left_thumb, 22=right_thumb
        landmark_ids = {
            'left_wrist': 15,
            'right_wrist': 16,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
        }

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            if not keypoints:
                continue

            kp_dict = {kp['id']: kp for kp in keypoints}

            # Check for wrist landmarks (minimum requirement)
            if not all(k in kp_dict for k in [15, 16]):
                continue

            ts = frame.get('timestamp', frame.get('frame', 0) / self.fps)
            trajectories['timestamps'].append(ts)

            for name, lid in landmark_ids.items():
                if lid in kp_dict:
                    trajectories[name].append([
                        kp_dict[lid]['x'],
                        kp_dict[lid]['y'],
                        kp_dict[lid].get('z', 0)
                    ])
                else:
                    last_val = trajectories[name][-1] if trajectories[name] else [0.5, 0.5, 0]
                    trajectories[name].append(last_val)

        # Convert to numpy
        for key in trajectories:
            trajectories[key] = np.array(trajectories[key])

        return trajectories

    def _extract_trajectories(self, landmark_frames: List[Dict]) -> Dict:
        """Extract joint trajectories from frames for gait analysis"""
        trajectories = {
            'timestamps': [],
            'left_hip': [], 'right_hip': [],
            'left_knee': [], 'right_knee': [],
            'left_ankle': [], 'right_ankle': [],
            'left_shoulder': [], 'right_shoulder': [],
            'left_wrist': [], 'right_wrist': [],
        }

        landmark_ids = {
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_wrist': 15, 'right_wrist': 16,
        }

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            if not keypoints:
                continue

            kp_dict = {kp['id']: kp for kp in keypoints}

            # Check required landmarks
            if not all(k in kp_dict for k in [23, 24, 27, 28]):
                continue

            ts = frame.get('timestamp', frame.get('frame', 0) / self.fps)
            trajectories['timestamps'].append(ts)

            for name, lid in landmark_ids.items():
                if lid in kp_dict:
                    trajectories[name].append([
                        kp_dict[lid]['x'],
                        kp_dict[lid]['y'],
                        kp_dict[lid].get('z', 0)
                    ])
                else:
                    last_val = trajectories[name][-1] if trajectories[name] else [0.5, 0.5, 0]
                    trajectories[name].append(last_val)

        # Convert to numpy
        for key in trajectories:
            trajectories[key] = np.array(trajectories[key])

        return trajectories

    def _generate_finger_tapping_data(self, trajectories: Dict) -> Dict:
        """Generate visualization data specific to finger tapping"""
        return {
            "task_type": "finger_tapping",
            "joint_angles": self._generate_tapping_amplitude(trajectories),
            "symmetry": self._generate_finger_symmetry(trajectories),
            "gait_cycles": self._generate_tapping_rhythm(trajectories),
            "speed_profile": self._generate_tapping_speed(trajectories)
        }

    def _generate_gait_data(self, trajectories: Dict, gait_analysis: Optional[Dict]) -> Dict:
        """Generate visualization data specific to gait"""
        return {
            "task_type": "gait",
            "joint_angles": self._generate_joint_angles(trajectories),
            "symmetry": self._generate_symmetry_data(trajectories, gait_analysis),
            "gait_cycles": self._generate_gait_cycle_data(gait_analysis),
            "speed_profile": self._generate_speed_profile(trajectories)
        }

    # ==================== FINGER TAPPING SPECIFIC METHODS ====================

    def _generate_tapping_amplitude(self, trajectories: Dict) -> List[Dict]:
        """Generate tapping amplitude over time for AmplitudeChart"""
        if len(trajectories['timestamps']) < 10:
            return []

        timestamps = trajectories['timestamps']
        amplitude_data = []

        # Calculate amplitude using index finger or wrist Y position
        left_hand = trajectories.get('left_index', trajectories['left_wrist'])
        right_hand = trajectories.get('right_index', trajectories['right_wrist'])

        # Calculate rolling amplitude (peak-to-peak in sliding window)
        window_size = max(5, int(self.fps * 0.2))  # 200ms window

        for i in range(window_size, len(timestamps)):
            left_window = left_hand[i-window_size:i, 1]  # Y axis
            right_window = right_hand[i-window_size:i, 1]

            left_amp = np.ptp(left_window) * 100  # Scale to percentage
            right_amp = np.ptp(right_window) * 100

            amplitude_data.append({
                "frame": i,
                "time": round(float(timestamps[i]), 2),
                "leftAmplitude": round(float(left_amp), 1),
                "rightAmplitude": round(float(right_amp), 1),
                "avgAmplitude": round(float((left_amp + right_amp) / 2), 1)
            })

        # Subsample if too many frames
        if len(amplitude_data) > 200:
            step = len(amplitude_data) // 200
            amplitude_data = amplitude_data[::step]

        return amplitude_data

    def _generate_finger_symmetry(self, trajectories: Dict) -> List[Dict]:
        """Generate left/right hand symmetry data for finger tapping"""
        if len(trajectories['timestamps']) < 30:
            return []

        symmetry_data = []
        timestamps = trajectories['timestamps']
        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        # Use wrist or index finger positions
        left_hand = trajectories.get('left_index', trajectories['left_wrist'])
        right_hand = trajectories.get('right_index', trajectories['right_wrist'])

        # 1. Velocity symmetry (tapping speed)
        if len(left_hand) > 1:
            left_vel = np.linalg.norm(np.diff(left_hand, axis=0), axis=1) / dt
            right_vel = np.linalg.norm(np.diff(right_hand, axis=0), axis=1) / dt

            left_vel_med = np.median(left_vel)
            right_vel_med = np.median(right_vel)
            total_vel = left_vel_med + right_vel_med + 0.001

            symmetry_data.append({
                "metric": "탭핑 속도",
                "left": round(left_vel_med / total_vel * 200, 1),
                "right": round(right_vel_med / total_vel * 200, 1),
                "normal": 100
            })

        # 2. Amplitude symmetry
        left_amp = np.ptp(left_hand[:, 1])  # Y-axis range
        right_amp = np.ptp(right_hand[:, 1])
        total_amp = left_amp + right_amp + 0.001

        symmetry_data.append({
            "metric": "진폭",
            "left": round(left_amp / total_amp * 200, 1),
            "right": round(right_amp / total_amp * 200, 1),
            "normal": 100
        })

        # 3. Rhythm regularity
        # Detect tapping peaks for each hand
        left_vel_smooth = savgol_filter(left_vel, min(15, len(left_vel) // 2 * 2 + 1), 3) if len(left_vel) > 15 else left_vel
        right_vel_smooth = savgol_filter(right_vel, min(15, len(right_vel) // 2 * 2 + 1), 3) if len(right_vel) > 15 else right_vel

        left_peaks, _ = find_peaks(left_vel_smooth, distance=int(self.fps * 0.15))
        right_peaks, _ = find_peaks(right_vel_smooth, distance=int(self.fps * 0.15))

        if len(left_peaks) > 2:
            left_intervals = np.diff(timestamps[left_peaks])
            left_regularity = 1.0 - min(1.0, np.std(left_intervals) / (np.mean(left_intervals) + 0.001))
        else:
            left_regularity = 0.5

        if len(right_peaks) > 2:
            right_intervals = np.diff(timestamps[right_peaks])
            right_regularity = 1.0 - min(1.0, np.std(right_intervals) / (np.mean(right_intervals) + 0.001))
        else:
            right_regularity = 0.5

        total_reg = left_regularity + right_regularity + 0.001
        symmetry_data.append({
            "metric": "리듬 규칙성",
            "left": round(left_regularity / total_reg * 200, 1),
            "right": round(right_regularity / total_reg * 200, 1),
            "normal": 100
        })

        return symmetry_data

    def _generate_tapping_rhythm(self, trajectories: Dict) -> List[Dict]:
        """Generate tapping rhythm/interval data for RhythmChart"""
        if len(trajectories['timestamps']) < 30:
            return []

        timestamps = trajectories['timestamps']
        rhythm_data = []

        # Use wrist velocity to detect taps
        left_hand = trajectories.get('left_index', trajectories['left_wrist'])
        right_hand = trajectories.get('right_index', trajectories['right_wrist'])

        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        left_vel = np.linalg.norm(np.diff(left_hand, axis=0), axis=1) / dt
        right_vel = np.linalg.norm(np.diff(right_hand, axis=0), axis=1) / dt

        # Smooth velocities
        if len(left_vel) > 15:
            left_vel = savgol_filter(left_vel, min(15, len(left_vel) // 2 * 2 + 1), 3)
            right_vel = savgol_filter(right_vel, min(15, len(right_vel) // 2 * 2 + 1), 3)

        # Detect peaks (taps)
        left_peaks, left_props = find_peaks(left_vel, distance=int(self.fps * 0.15), height=np.median(left_vel))
        right_peaks, right_props = find_peaks(right_vel, distance=int(self.fps * 0.15), height=np.median(right_vel))

        # Create rhythm data entries
        tap_count = 0
        for i, peak in enumerate(left_peaks):
            if i > 0:
                interval = (timestamps[peak] - timestamps[left_peaks[i-1]]) * 1000  # ms
            else:
                interval = 0
            tap_count += 1
            rhythm_data.append({
                "tap": tap_count,
                "time": round(float(timestamps[peak]), 2),
                "interval": round(interval, 0),
                "side": "L",
                "amplitude": round(float(left_props['peak_heights'][i]) * 100, 1) if 'peak_heights' in left_props else 0
            })

        for i, peak in enumerate(right_peaks):
            if i > 0:
                interval = (timestamps[peak] - timestamps[right_peaks[i-1]]) * 1000  # ms
            else:
                interval = 0
            tap_count += 1
            rhythm_data.append({
                "tap": tap_count,
                "time": round(float(timestamps[peak]), 2),
                "interval": round(interval, 0),
                "side": "R",
                "amplitude": round(float(right_props['peak_heights'][i]) * 100, 1) if 'peak_heights' in right_props else 0
            })

        # Sort by time and renumber
        rhythm_data.sort(key=lambda x: x['time'])
        for i, item in enumerate(rhythm_data):
            item['tap'] = i + 1

        return rhythm_data[:50]  # Limit to 50 taps

    def _generate_tapping_speed(self, trajectories: Dict) -> List[Dict]:
        """Generate tapping speed profile over time"""
        if len(trajectories['timestamps']) < 30:
            return []

        timestamps = trajectories['timestamps']
        speed_data = []

        left_hand = trajectories.get('left_index', trajectories['left_wrist'])
        right_hand = trajectories.get('right_index', trajectories['right_wrist'])

        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        # Calculate instantaneous speeds
        left_vel = np.linalg.norm(np.diff(left_hand, axis=0), axis=1) / dt
        right_vel = np.linalg.norm(np.diff(right_hand, axis=0), axis=1) / dt

        # Smooth
        if len(left_vel) > 15:
            left_vel = savgol_filter(left_vel, min(15, len(left_vel) // 2 * 2 + 1), 3)
            right_vel = savgol_filter(right_vel, min(15, len(right_vel) // 2 * 2 + 1), 3)

        for i in range(len(left_vel)):
            speed_data.append({
                "time": round(float(timestamps[i+1]), 2),
                "leftSpeed": round(float(left_vel[i]) * 10, 2),  # Scale for visibility
                "rightSpeed": round(float(right_vel[i]) * 10, 2),
                "avgSpeed": round(float((left_vel[i] + right_vel[i]) / 2) * 10, 2)
            })

        # Subsample if too many points
        if len(speed_data) > 100:
            step = len(speed_data) // 100
            speed_data = speed_data[::step]

        return speed_data

    # ==================== GAIT SPECIFIC METHODS ====================

    def _generate_joint_angles(self, trajectories: Dict) -> List[Dict]:
        """Generate joint angle data for JointAngleChart"""
        if len(trajectories['timestamps']) < 10:
            return []

        timestamps = trajectories['timestamps']
        joint_angles = []

        for i in range(len(timestamps)):
            # Calculate knee angles (angle between thigh and shank)
            left_knee_angle = self._calculate_knee_angle(
                trajectories['left_hip'][i],
                trajectories['left_knee'][i],
                trajectories['left_ankle'][i]
            )
            right_knee_angle = self._calculate_knee_angle(
                trajectories['right_hip'][i],
                trajectories['right_knee'][i],
                trajectories['right_ankle'][i]
            )

            # Calculate hip flexion (angle from vertical)
            left_hip_angle = self._calculate_hip_angle(
                trajectories['left_shoulder'][i] if len(trajectories['left_shoulder']) > i else None,
                trajectories['left_hip'][i],
                trajectories['left_knee'][i]
            )
            right_hip_angle = self._calculate_hip_angle(
                trajectories['right_shoulder'][i] if len(trajectories['right_shoulder']) > i else None,
                trajectories['right_hip'][i],
                trajectories['right_knee'][i]
            )

            joint_angles.append({
                "frame": i,
                "time": round(float(timestamps[i]), 2),
                "leftKnee": round(left_knee_angle, 1),
                "rightKnee": round(right_knee_angle, 1),
                "leftHip": round(left_hip_angle, 1),
                "rightHip": round(right_hip_angle, 1)
            })

        # Subsample if too many frames (max 200 points for chart)
        if len(joint_angles) > 200:
            step = len(joint_angles) // 200
            joint_angles = joint_angles[::step]

        return joint_angles

    def _calculate_knee_angle(self, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """Calculate knee flexion angle"""
        try:
            hip = np.array(hip)
            knee = np.array(knee)
            ankle = np.array(ankle)

            thigh = hip - knee
            shank = ankle - knee

            cos_angle = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            # Convert to flexion angle (180 - angle)
            flexion = 180 - angle
            return max(0, min(90, flexion))  # Clamp to reasonable range
        except:
            return 0.0

    def _calculate_hip_angle(self, shoulder: Optional[np.ndarray], hip: np.ndarray, knee: np.ndarray) -> float:
        """Calculate hip flexion angle from vertical"""
        try:
            hip = np.array(hip)
            knee = np.array(knee)

            thigh = knee - hip
            vertical = np.array([0, 1, 0])  # Y-down in normalized coords

            cos_angle = np.dot(thigh, vertical) / (np.linalg.norm(thigh) * np.linalg.norm(vertical) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            return max(0, min(60, angle))  # Clamp to reasonable range
        except:
            return 0.0

    def _generate_symmetry_data(self, trajectories: Dict, gait_analysis: Optional[Dict]) -> List[Dict]:
        """Generate symmetry comparison data for SymmetryChart - IMPROVED VERSION"""
        if len(trajectories['timestamps']) < 30:
            return []

        symmetry_data = []

        # Use gait_analysis if available (most accurate)
        if gait_analysis:
            asym = gait_analysis.get('asymmetry_percent', {})

            # Step length asymmetry
            step_asym = asym.get('step_length', 0)
            left_step = 100 - step_asym / 2
            right_step = 100 + step_asym / 2
            symmetry_data.append({
                "metric": "보폭",
                "left": round(left_step, 1),
                "right": round(right_step, 1),
                "normal": 100
            })

            # Swing time asymmetry
            swing_asym = asym.get('swing_time', 0)
            left_swing = 100 - swing_asym / 2
            right_swing = 100 + swing_asym / 2
            symmetry_data.append({
                "metric": "스윙 시간",
                "left": round(left_swing, 1),
                "right": round(right_swing, 1),
                "normal": 100
            })

            # Stance time asymmetry
            stance_asym = asym.get('stance_time', 0)
            left_stance = 100 - stance_asym / 2
            right_stance = 100 + stance_asym / 2
            symmetry_data.append({
                "metric": "입각기",
                "left": round(left_stance, 1),
                "right": round(right_stance, 1),
                "normal": 100
            })

        # Calculate velocity-based symmetry (more reliable than position)
        timestamps = trajectories['timestamps']
        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        # Left/Right ankle velocity magnitude
        left_ankle = trajectories['left_ankle']
        right_ankle = trajectories['right_ankle']

        if len(left_ankle) > 1:
            left_vel = np.linalg.norm(np.diff(left_ankle, axis=0), axis=1) / dt
            right_vel = np.linalg.norm(np.diff(right_ankle, axis=0), axis=1) / dt

            # Use median to avoid outliers
            left_vel_med = np.median(left_vel)
            right_vel_med = np.median(right_vel)
            total_vel = left_vel_med + right_vel_med + 0.001

            symmetry_data.append({
                "metric": "발 속도",
                "left": round(left_vel_med / total_vel * 200, 1),
                "right": round(right_vel_med / total_vel * 200, 1),
                "normal": 100
            })

        # Knee ROM symmetry (per-frame calculation averaged)
        left_knee_angles = []
        right_knee_angles = []
        for i in range(len(timestamps)):
            left_knee_angles.append(self._calculate_knee_angle(
                trajectories['left_hip'][i],
                trajectories['left_knee'][i],
                trajectories['left_ankle'][i]
            ))
            right_knee_angles.append(self._calculate_knee_angle(
                trajectories['right_hip'][i],
                trajectories['right_knee'][i],
                trajectories['right_ankle'][i]
            ))

        # Use interquartile range for robustness
        left_knee_iqr = np.percentile(left_knee_angles, 75) - np.percentile(left_knee_angles, 25)
        right_knee_iqr = np.percentile(right_knee_angles, 75) - np.percentile(right_knee_angles, 25)
        total_knee = left_knee_iqr + right_knee_iqr + 0.001

        symmetry_data.append({
            "metric": "무릎 굴곡",
            "left": round(left_knee_iqr / total_knee * 200, 1),
            "right": round(right_knee_iqr / total_knee * 200, 1),
            "normal": 100
        })

        # Hip ROM symmetry
        left_hip_angles = []
        right_hip_angles = []
        for i in range(len(timestamps)):
            left_hip_angles.append(self._calculate_hip_angle(
                None, trajectories['left_hip'][i], trajectories['left_knee'][i]
            ))
            right_hip_angles.append(self._calculate_hip_angle(
                None, trajectories['right_hip'][i], trajectories['right_knee'][i]
            ))

        left_hip_iqr = np.percentile(left_hip_angles, 75) - np.percentile(left_hip_angles, 25)
        right_hip_iqr = np.percentile(right_hip_angles, 75) - np.percentile(right_hip_angles, 25)
        total_hip = left_hip_iqr + right_hip_iqr + 0.001

        symmetry_data.append({
            "metric": "엉덩이 굴곡",
            "left": round(left_hip_iqr / total_hip * 200, 1),
            "right": round(right_hip_iqr / total_hip * 200, 1),
            "normal": 100
        })

        # Arm swing symmetry (velocity-based)
        left_wrist = trajectories['left_wrist']
        right_wrist = trajectories['right_wrist']

        if len(left_wrist) > 1:
            left_arm_vel = np.linalg.norm(np.diff(left_wrist, axis=0), axis=1) / dt
            right_arm_vel = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1) / dt

            left_arm_med = np.median(left_arm_vel)
            right_arm_med = np.median(right_arm_vel)
            total_arm = left_arm_med + right_arm_med + 0.001

            symmetry_data.append({
                "metric": "팔 흔들기",
                "left": round(left_arm_med / total_arm * 200, 1),
                "right": round(right_arm_med / total_arm * 200, 1),
                "normal": 100
            })

        return symmetry_data

    def _generate_gait_cycle_data(self, gait_analysis: Optional[Dict]) -> List[Dict]:
        """Generate gait cycle timing data for GaitCycleChart"""
        if not gait_analysis:
            return []

        cycles = gait_analysis.get('cycles', {})
        left_cycles = cycles.get('left', [])
        right_cycles = cycles.get('right', [])

        all_cycles = []

        # Interleave left and right cycles by time
        for i, cycle in enumerate(left_cycles):
            durations = cycle.get('durations_sec', {})
            phase = cycle.get('phase_percent', {})
            all_cycles.append({
                "step": len(all_cycles) + 1,
                "duration": round(durations.get('total', 1.0) * 1000),  # Convert to ms
                "stancePhase": round(phase.get('stance', 60)),
                "swingPhase": round(phase.get('swing', 40)),
                "strideLength": round(cycle.get('spatial_meters', {}).get('step_length', 0.7), 2),
                "side": "L"
            })

        for i, cycle in enumerate(right_cycles):
            durations = cycle.get('durations_sec', {})
            phase = cycle.get('phase_percent', {})
            all_cycles.append({
                "step": len(all_cycles) + 1,
                "duration": round(durations.get('total', 1.0) * 1000),
                "stancePhase": round(phase.get('stance', 60)),
                "swingPhase": round(phase.get('swing', 40)),
                "strideLength": round(cycle.get('spatial_meters', {}).get('step_length', 0.7), 2),
                "side": "R"
            })

        # Sort by step number (re-number after merge)
        all_cycles.sort(key=lambda x: x['step'])
        for i, cycle in enumerate(all_cycles):
            cycle['step'] = i + 1

        return all_cycles[:20]  # Limit to 20 cycles

    def _generate_speed_profile(self, trajectories: Dict) -> List[Dict]:
        """Generate walking speed profile for SpeedProfileChart"""
        if len(trajectories['timestamps']) < 30:
            return []

        timestamps = trajectories['timestamps']

        # Calculate hip center position
        hip_center = (trajectories['left_hip'] + trajectories['right_hip']) / 2

        # Calculate instantaneous speed (displacement per frame)
        speeds = []
        for i in range(1, len(hip_center)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / self.fps

            # Displacement in Z direction (forward)
            displacement = np.linalg.norm(hip_center[i] - hip_center[i-1])

            # Scale to approximate m/s (using hip width scaling)
            left_hip = trajectories['left_hip']
            right_hip = trajectories['right_hip']
            hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
            avg_hip_width_norm = np.mean(hip_widths[hip_widths > 0.01])

            REAL_HIP_WIDTH = 0.30  # meters
            scale = REAL_HIP_WIDTH / avg_hip_width_norm if avg_hip_width_norm > 0.01 else 2.0
            scale = np.clip(scale, 1.0, 10.0)

            speed_mps = (displacement * scale) / dt
            speeds.append(speed_mps)

        # Smooth the speed signal
        if len(speeds) > 15:
            speeds = savgol_filter(speeds, min(15, len(speeds) // 2 * 2 + 1), 3)

        # Generate output
        speed_profile = []
        for i, speed in enumerate(speeds):
            speed_profile.append({
                "time": round(float(timestamps[i+1]), 2),
                "speed": round(float(np.clip(speed, 0, 3.0)), 3),  # Clamp to reasonable range
                "normalLow": 0.8,
                "normalHigh": 1.2
            })

        # Subsample if too many points
        if len(speed_profile) > 100:
            step = len(speed_profile) // 100
            speed_profile = speed_profile[::step]

        return speed_profile


class EventDetector:
    """Detect clinically relevant events from gait/movement data - Enhanced Version with Population Stats"""

    def __init__(self, fps: float = 30.0, min_confidence: float = 0.5, use_population_stats: bool = True):
        self.fps = fps
        self.min_confidence = min_confidence  # Minimum confidence to report event
        self.baseline = {}  # Adaptive baseline from early video
        self.use_population_stats = use_population_stats
        self.population_stats = None

        # Load population stats if enabled
        if use_population_stats:
            try:
                self.population_stats = PopulationStatsLoader.get_instance()
            except Exception as e:
                print(f"Warning: Could not load population stats: {e}")
                self.use_population_stats = False

    def detect_events(self, trajectories: Dict, gait_analysis: Optional[Dict] = None, task_type: str = "gait") -> List[Dict]:
        """
        Detect events based on task type with confidence scoring

        Args:
            trajectories: Joint trajectory data
            gait_analysis: Pre-computed gait analysis
            task_type: 'gait' or 'finger_tapping'

        Returns:
            List of detected events with timestamp, type, description, confidence
        """
        events = []

        # Step 1: Calculate adaptive baseline from early frames (first 1-2 seconds)
        self._calculate_baseline(trajectories, task_type)

        if task_type == "gait":
            events.extend(self._detect_speed_drops(trajectories))
            events.extend(self._detect_pauses(trajectories))
            events.extend(self._detect_asymmetry_events(trajectories, gait_analysis))
        elif task_type == "finger_tapping":
            events.extend(self._detect_fatigue(trajectories))
            events.extend(self._detect_finger_pauses(trajectories))
            events.extend(self._detect_rhythm_breaks(trajectories))

        # Step 2: Filter by confidence
        events = [e for e in events if e.get('confidence', 1.0) >= self.min_confidence]

        # Step 3: Temporal filtering - merge nearby events of same type
        events = self._merge_nearby_events(events, min_gap=0.5)

        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])

        return events

    def _calculate_baseline(self, trajectories: Dict, task_type: str):
        """Calculate adaptive baseline from early video frames (first 1-2 seconds)"""
        if len(trajectories.get('timestamps', [])) < 30:
            return

        timestamps = trajectories['timestamps']
        baseline_end = min(int(self.fps * 1.5), len(timestamps) // 3)  # First 1.5s or 1/3 of video

        if task_type == "finger_tapping":
            left_hand = trajectories.get('left_index', trajectories.get('left_wrist', np.array([])))
            right_hand = trajectories.get('right_index', trajectories.get('right_wrist', np.array([])))

            if len(left_hand) > baseline_end and len(right_hand) > baseline_end:
                dt = np.diff(timestamps[:baseline_end])
                dt[dt <= 0] = 1.0 / self.fps

                left_vel = np.linalg.norm(np.diff(left_hand[:baseline_end], axis=0), axis=1) / dt
                right_vel = np.linalg.norm(np.diff(right_hand[:baseline_end], axis=0), axis=1) / dt

                self.baseline['velocity_median'] = np.median(np.concatenate([left_vel, right_vel]))
                self.baseline['velocity_std'] = np.std(np.concatenate([left_vel, right_vel]))
                self.baseline['amplitude_left'] = np.ptp(left_hand[:baseline_end, 1])
                self.baseline['amplitude_right'] = np.ptp(right_hand[:baseline_end, 1])

        else:  # gait
            left_hip = trajectories.get('left_hip', np.array([]))
            right_hip = trajectories.get('right_hip', np.array([]))

            if len(left_hip) > baseline_end:
                hip_center = (left_hip[:baseline_end] + right_hip[:baseline_end]) / 2
                dt = np.diff(timestamps[:baseline_end])
                dt[dt <= 0] = 1.0 / self.fps

                speeds = np.linalg.norm(np.diff(hip_center, axis=0), axis=1) / dt
                self.baseline['speed_median'] = np.median(speeds)
                self.baseline['speed_std'] = np.std(speeds)

    def _merge_nearby_events(self, events: List[Dict], min_gap: float = 0.5) -> List[Dict]:
        """Merge events of the same type that are close in time"""
        if len(events) <= 1:
            return events

        # Sort by type, then by timestamp
        events.sort(key=lambda x: (x['type'], x['timestamp']))

        merged = []
        i = 0

        while i < len(events):
            current = events[i].copy()
            j = i + 1

            # Find consecutive events of same type within min_gap
            while j < len(events):
                if events[j]['type'] != current['type']:
                    break
                if events[j]['timestamp'] - events[j-1]['timestamp'] > min_gap:
                    break

                # Merge: keep highest confidence, update description
                if events[j].get('confidence', 0) > current.get('confidence', 0):
                    current['confidence'] = events[j]['confidence']
                j += 1

            # If multiple events were merged, update description
            if j - i > 1:
                current['description'] += f" (x{j-i})"

            merged.append(current)
            i = j

        return merged

    def _calculate_confidence(self, deviation_ratio: float, min_ratio: float = 0.3, max_ratio: float = 1.0) -> float:
        """
        Calculate confidence score based on deviation from baseline

        Args:
            deviation_ratio: How much the value deviates (e.g., 0.5 = 50% deviation)
            min_ratio: Minimum deviation for non-zero confidence
            max_ratio: Deviation for maximum confidence

        Returns:
            Confidence score between 0 and 1
        """
        if deviation_ratio < min_ratio:
            return 0.0
        if deviation_ratio >= max_ratio:
            return 1.0

        # Linear interpolation
        return (deviation_ratio - min_ratio) / (max_ratio - min_ratio)

    def _detect_speed_drops(self, trajectories: Dict) -> List[Dict]:
        """Detect sudden speed drops with adaptive threshold and confidence"""
        events = []

        if len(trajectories['timestamps']) < 30:
            return events

        timestamps = trajectories['timestamps']
        hip_center = (trajectories['left_hip'] + trajectories['right_hip']) / 2

        # Calculate speeds
        speeds = []
        for i in range(1, len(hip_center)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / self.fps
            displacement = np.linalg.norm(hip_center[i] - hip_center[i-1])
            speeds.append(displacement / dt)

        if len(speeds) < 10:
            return events

        # Smooth speeds
        speeds = savgol_filter(speeds, min(15, len(speeds) // 2 * 2 + 1), 3)

        # Use adaptive baseline if available, else use median
        baseline_speed = self.baseline.get('speed_median', np.median(speeds))
        baseline_std = self.baseline.get('speed_std', np.std(speeds))

        # Adaptive threshold: minimum 20%, maximum 50% drop
        min_drop_threshold = 20  # Minimum % drop to consider
        high_confidence_drop = 40  # % drop for high confidence

        window = int(self.fps * 0.5)  # 0.5 second window
        i = window

        while i < len(speeds) - window:
            prev_avg = np.mean(speeds[i-window:i])
            curr_avg = np.mean(speeds[i:i+window])

            if prev_avg > 0.01:  # Avoid division by zero
                drop_percent = (prev_avg - curr_avg) / prev_avg * 100

                if drop_percent > min_drop_threshold:
                    # Calculate confidence based on drop magnitude
                    confidence = self._calculate_confidence(
                        drop_percent / 100,
                        min_ratio=min_drop_threshold / 100,
                        max_ratio=high_confidence_drop / 100
                    )

                    events.append({
                        "timestamp": round(float(timestamps[i+1]), 2),
                        "type": "speed_drop",
                        "description": f"속도 급감 ({int(drop_percent)}% 감소)",
                        "confidence": round(confidence, 2)
                    })
                    # Skip ahead to avoid duplicate detections
                    i += window
            i += 1

        return events[:5]  # Limit to 5 events

    def _detect_pauses(self, trajectories: Dict) -> List[Dict]:
        """Detect movement pauses with adaptive threshold and confidence - Gait"""
        events = []

        if len(trajectories['timestamps']) < 30:
            return events

        timestamps = trajectories['timestamps']
        hip_center = (trajectories['left_hip'] + trajectories['right_hip']) / 2

        # Calculate speeds
        speeds = []
        for i in range(1, len(hip_center)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / self.fps
            displacement = np.linalg.norm(hip_center[i] - hip_center[i-1])
            speeds.append(displacement / dt)

        if len(speeds) < 10:
            return events

        # Use adaptive baseline if available
        baseline_speed = self.baseline.get('speed_median', np.median(speeds))

        # Adaptive threshold based on baseline
        pause_threshold = baseline_speed * 0.15  # 15% of baseline = pause
        min_pause_duration = 0.4  # At least 0.4 seconds
        high_confidence_duration = 1.0  # 1 second pause = high confidence
        min_pause_frames = int(self.fps * min_pause_duration)

        pause_start = None
        pause_count = 0

        for i, speed in enumerate(speeds):
            if speed < pause_threshold:
                if pause_start is None:
                    pause_start = i
                pause_count += 1
            else:
                if pause_count >= min_pause_frames and pause_start is not None:
                    pause_duration = pause_count / self.fps

                    # Confidence based on pause duration
                    confidence = self._calculate_confidence(
                        pause_duration,
                        min_ratio=min_pause_duration,
                        max_ratio=high_confidence_duration
                    )

                    events.append({
                        "timestamp": round(float(timestamps[pause_start+1]), 2),
                        "type": "pause",
                        "description": f"멈춤 감지 ({pause_duration:.1f}초)",
                        "confidence": round(confidence, 2)
                    })
                pause_start = None
                pause_count = 0

        return events[:5]  # Limit to 5 events

    def _detect_finger_pauses(self, trajectories: Dict) -> List[Dict]:
        """Detect tapping pauses with adaptive threshold and confidence"""
        events = []

        if len(trajectories['timestamps']) < 30:
            return events

        timestamps = trajectories['timestamps']
        left_hand = trajectories.get('left_index', trajectories.get('left_wrist', np.array([])))
        right_hand = trajectories.get('right_index', trajectories.get('right_wrist', np.array([])))

        if len(left_hand) < 2 or len(right_hand) < 2:
            return events

        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        # Calculate combined hand velocities
        left_vel = np.linalg.norm(np.diff(left_hand, axis=0), axis=1) / dt
        right_vel = np.linalg.norm(np.diff(right_hand, axis=0), axis=1) / dt
        combined_vel = (left_vel + right_vel) / 2

        # Use adaptive baseline if available
        baseline = self.baseline.get('velocity_median', np.median(combined_vel))

        # Adaptive thresholds
        pause_threshold = baseline * 0.2  # 20% of baseline (more lenient)
        min_pause_duration = 0.3  # At least 0.3 seconds
        high_confidence_duration = 0.8  # 0.8 second pause = high confidence
        min_pause_frames = int(self.fps * min_pause_duration)

        pause_start = None
        pause_count = 0

        for i, vel in enumerate(combined_vel):
            if vel < pause_threshold:
                if pause_start is None:
                    pause_start = i
                pause_count += 1
            else:
                if pause_count >= min_pause_frames and pause_start is not None:
                    pause_duration = pause_count / self.fps

                    # Confidence based on pause duration
                    confidence = self._calculate_confidence(
                        pause_duration,
                        min_ratio=min_pause_duration,
                        max_ratio=high_confidence_duration
                    )

                    events.append({
                        "timestamp": round(float(timestamps[pause_start+1]), 2),
                        "type": "pause",
                        "description": f"탭핑 멈춤 ({pause_duration:.1f}초)",
                        "confidence": round(confidence, 2)
                    })
                pause_start = None
                pause_count = 0

        return events[:5]

    def _detect_fatigue(self, trajectories: Dict) -> List[Dict]:
        """Detect amplitude fatigue with adaptive threshold and confidence (uses population stats)"""
        events = []

        if len(trajectories['timestamps']) < 60:  # Need enough data
            return events

        timestamps = trajectories['timestamps']

        # Use wrist movement for finger tapping
        left_wrist = trajectories.get('left_wrist', np.array([]))
        right_wrist = trajectories.get('right_wrist', np.array([]))

        if len(left_wrist) < 30 or len(right_wrist) < 30:
            return events

        # Calculate amplitude over time (sliding window)
        window_frames = int(self.fps * 1.0)  # 1 second window

        # Get thresholds from population stats if available
        if self.use_population_stats and self.population_stats:
            stats = self.population_stats.get_stats('finger_tapping')
            # Use p75 as the "moderate" threshold and p90 as "high confidence"
            min_fatigue_percent = stats.get('amplitude_decrement_median', 10.0)
            high_confidence_fatigue = stats.get('amplitude_decrement_p75', 20.0)
        else:
            # Fallback to hardcoded thresholds
            min_fatigue_percent = 20  # Minimum 20% decrease to consider
            high_confidence_fatigue = 40  # 40% decrease = high confidence

        # Check both hands
        for wrist, side in [(left_wrist, "왼손"), (right_wrist, "오른손")]:
            if len(wrist) < window_frames * 3:
                continue

            amplitudes = []
            for i in range(0, len(wrist) - window_frames, window_frames // 2):
                window_data = wrist[i:i+window_frames]
                amp = np.ptp(window_data[:, 1])  # Y-axis range
                amplitudes.append(amp)

            if len(amplitudes) < 3:
                continue

            # Compare first third vs last third
            first_third = np.mean(amplitudes[:len(amplitudes)//3])
            last_third = np.mean(amplitudes[-len(amplitudes)//3:])

            if first_third > 0.01:
                fatigue_percent = (first_third - last_third) / first_third * 100

                if fatigue_percent > min_fatigue_percent:
                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        fatigue_percent / 100,
                        min_ratio=min_fatigue_percent / 100,
                        max_ratio=high_confidence_fatigue / 100
                    )

                    events.append({
                        "timestamp": round(float(timestamps[len(timestamps)//2]), 2),
                        "type": "fatigue",
                        "description": f"피로 감지 - {side} 진폭 {int(fatigue_percent)}% 감소",
                        "confidence": round(confidence, 2)
                    })

        return events

    def _detect_rhythm_breaks(self, trajectories: Dict) -> List[Dict]:
        """Detect rhythm irregularities with adaptive threshold and confidence (uses population stats)"""
        events = []

        if len(trajectories['timestamps']) < 60:
            return events

        timestamps = trajectories['timestamps']
        left_hand = trajectories.get('left_index', trajectories.get('left_wrist', np.array([])))
        right_hand = trajectories.get('right_index', trajectories.get('right_wrist', np.array([])))

        if len(left_hand) < 30:
            return events

        dt = np.diff(timestamps)
        dt[dt <= 0] = 1.0 / self.fps

        # Get thresholds from population stats if available
        if self.use_population_stats and self.population_stats:
            stats = self.population_stats.get_stats('finger_tapping')
            # Rhythm variability from population data
            # Convert to deviation ratio (rhythm_variability is in %, we need ratio)
            median_variability = stats.get('rhythm_variability_median', 15.0)
            p75_variability = stats.get('rhythm_variability_p75', 25.0)
            # A rhythm break is when local variability exceeds the p75 of population
            min_deviation = median_variability / 100.0  # Convert to ratio
            high_confidence_deviation = p75_variability / 100.0
        else:
            # Fallback to hardcoded thresholds
            min_deviation = 0.4  # 40% deviation minimum
            high_confidence_deviation = 0.8  # 80% deviation = high confidence

        # Analyze both hands
        for hand, side in [(left_hand, "왼손"), (right_hand, "오른손")]:
            if len(hand) < 30:
                continue

            vel = np.linalg.norm(np.diff(hand, axis=0), axis=1) / dt

            if len(vel) > 15:
                vel = savgol_filter(vel, min(15, len(vel) // 2 * 2 + 1), 3)

            # Detect peaks (taps)
            peaks, _ = find_peaks(vel, distance=int(self.fps * 0.15), height=np.median(vel))

            if len(peaks) < 5:
                continue

            # Calculate intervals
            intervals = np.diff(timestamps[peaks])
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            # Find rhythm breaks (deviation from mean)
            for i, interval in enumerate(intervals):
                deviation_ratio = abs(interval - mean_interval) / (mean_interval + 0.001)

                if deviation_ratio > min_deviation:
                    # Calculate confidence based on deviation magnitude
                    confidence = self._calculate_confidence(
                        deviation_ratio,
                        min_ratio=min_deviation,
                        max_ratio=high_confidence_deviation
                    )

                    events.append({
                        "timestamp": round(float(timestamps[peaks[i+1]]), 2),
                        "type": "rhythm_break",
                        "description": f"{side} 리듬 불규칙 ({interval*1000:.0f}ms, 평균 {mean_interval*1000:.0f}ms)",
                        "confidence": round(confidence, 2)
                    })

        return events[:5]

    def _detect_asymmetry_events(self, trajectories: Dict, gait_analysis: Optional[Dict]) -> List[Dict]:
        """Detect significant asymmetry events with confidence (uses population stats)"""
        events = []

        # Get thresholds from population stats if available
        if self.use_population_stats and self.population_stats:
            stats = self.population_stats.get_stats('gait')
            # Use median as minimum and p75 as "high confidence"
            min_asymmetry = stats.get('step_length_asymmetry_median', 5.0)
            high_confidence_asymmetry = stats.get('step_length_asymmetry_p75', 10.0)
        else:
            # Fallback to hardcoded thresholds
            min_asymmetry = 10  # 10% minimum to consider
            high_confidence_asymmetry = 25  # 25% = high confidence

        if gait_analysis:
            asym = gait_analysis.get('asymmetry_percent', {})

            # Check step length asymmetry
            step_asym = abs(asym.get('step_length', 0))
            if step_asym > min_asymmetry:
                confidence = self._calculate_confidence(
                    step_asym / 100,
                    min_ratio=min_asymmetry / 100,
                    max_ratio=high_confidence_asymmetry / 100
                )
                events.append({
                    "timestamp": 0.0,
                    "type": "asymmetry",
                    "description": f"보폭 비대칭 {int(step_asym)}%",
                    "confidence": round(confidence, 2)
                })

            # Check swing time asymmetry (use separate thresholds if available)
            swing_asym = abs(asym.get('swing_time', 0))
            if self.use_population_stats and self.population_stats:
                stats = self.population_stats.get_stats('gait')
                swing_min = stats.get('swing_time_asymmetry_median', min_asymmetry)
                swing_high = stats.get('swing_time_asymmetry_p75', high_confidence_asymmetry)
            else:
                swing_min = min_asymmetry
                swing_high = high_confidence_asymmetry

            if swing_asym > swing_min:
                confidence = self._calculate_confidence(
                    swing_asym / 100,
                    min_ratio=swing_min / 100,
                    max_ratio=swing_high / 100
                )
                events.append({
                    "timestamp": 0.0,
                    "type": "asymmetry",
                    "description": f"스윙 시간 비대칭 {int(swing_asym)}%",
                    "confidence": round(confidence, 2)
                })

        return events


def generate_visualization_data(landmark_frames: List[Dict],
                                gait_analysis: Optional[Dict] = None,
                                fps: float = 30.0,
                                task_type: str = "gait") -> Dict:
    """
    Convenience function to generate visualization data

    Args:
        landmark_frames: List of frame data with landmarks
        gait_analysis: Optional pre-computed gait analysis
        fps: Video frame rate
        task_type: 'gait' or 'finger_tapping'

    Returns:
        Dict with visualization data
    """
    generator = VisualizationDataGenerator(fps=fps)
    return generator.generate(landmark_frames, gait_analysis, task_type)


def detect_events(landmark_frames: List[Dict],
                  gait_analysis: Optional[Dict] = None,
                  fps: float = 30.0,
                  task_type: str = "gait",
                  use_population_stats: bool = True) -> List[Dict]:
    """
    Convenience function to detect events

    Args:
        landmark_frames: List of frame data with landmarks
        gait_analysis: Optional pre-computed gait analysis
        fps: Video frame rate
        task_type: 'gait' or 'finger_tapping'
        use_population_stats: Whether to use population-based thresholds from training data

    Returns:
        List of detected events
    """
    generator = VisualizationDataGenerator(fps=fps)

    if task_type == "finger_tapping":
        trajectories = generator._extract_finger_trajectories(landmark_frames)
    else:
        trajectories = generator._extract_trajectories(landmark_frames)

    detector = EventDetector(fps=fps, use_population_stats=use_population_stats)
    return detector.detect_events(trajectories, gait_analysis, task_type)


def get_population_stats_info(task_type: str = "gait") -> Dict:
    """
    Get information about the population statistics being used for event detection.
    Useful for debugging and transparency.

    Args:
        task_type: 'gait' or 'finger_tapping'

    Returns:
        Dict with threshold information
    """
    try:
        loader = PopulationStatsLoader.get_instance()
        stats = loader.get_stats(task_type)
        return {
            "source": "training_data" if stats else "defaults",
            "task_type": task_type,
            "thresholds": stats
        }
    except Exception as e:
        return {
            "source": "defaults",
            "task_type": task_type,
            "error": str(e),
            "thresholds": {}
        }
