"""
Visualization Data Generator
Generates data for frontend visualization charts from skeleton landmarks
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from scipy.signal import savgol_filter, find_peaks


@dataclass
class VisualizationData:
    """Container for all visualization chart data"""
    joint_angles: List[Dict]  # For JointAngleChart
    symmetry: List[Dict]      # For SymmetryChart
    gait_cycles: List[Dict]   # For GaitCycleChart
    speed_profile: List[Dict] # For SpeedProfileChart


class VisualizationDataGenerator:
    """Generate visualization data from skeleton landmarks"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def generate(self, landmark_frames: List[Dict], gait_analysis: Optional[Dict] = None) -> Dict:
        """
        Generate all visualization data from landmarks

        Args:
            landmark_frames: List of frame data with landmarks
            gait_analysis: Optional pre-computed gait cycle analysis

        Returns:
            Dict with visualization data for all charts
        """
        if len(landmark_frames) < 10:
            return self._empty_data()

        # Extract trajectories
        trajectories = self._extract_trajectories(landmark_frames)

        # Generate each chart's data
        joint_angles = self._generate_joint_angles(trajectories)
        symmetry = self._generate_symmetry_data(trajectories, gait_analysis)
        gait_cycles = self._generate_gait_cycle_data(gait_analysis)
        speed_profile = self._generate_speed_profile(trajectories)

        return {
            "joint_angles": joint_angles,
            "symmetry": symmetry,
            "gait_cycles": gait_cycles,
            "speed_profile": speed_profile
        }

    def _empty_data(self) -> Dict:
        """Return empty data structure"""
        return {
            "joint_angles": [],
            "symmetry": [],
            "gait_cycles": [],
            "speed_profile": []
        }

    def _extract_trajectories(self, landmark_frames: List[Dict]) -> Dict:
        """Extract joint trajectories from frames"""
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
    """Detect clinically relevant events from gait/movement data"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def detect_events(self, trajectories: Dict, gait_analysis: Optional[Dict] = None, task_type: str = "gait") -> List[Dict]:
        """
        Detect events based on task type

        Args:
            trajectories: Joint trajectory data
            gait_analysis: Pre-computed gait analysis
            task_type: 'gait' or 'finger_tapping'

        Returns:
            List of detected events with timestamp, type, description
        """
        events = []

        if task_type == "gait":
            events.extend(self._detect_speed_drops(trajectories))
            events.extend(self._detect_pauses(trajectories))
            events.extend(self._detect_asymmetry_events(trajectories, gait_analysis))
        elif task_type == "finger_tapping":
            events.extend(self._detect_fatigue(trajectories))
            events.extend(self._detect_pauses(trajectories))

        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])

        return events

    def _detect_speed_drops(self, trajectories: Dict) -> List[Dict]:
        """Detect sudden speed drops (>30% decrease)"""
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

        # Calculate rolling average
        window = int(self.fps * 0.5)  # 0.5 second window

        for i in range(window, len(speeds) - window):
            prev_avg = np.mean(speeds[i-window:i])
            curr_avg = np.mean(speeds[i:i+window])

            if prev_avg > 0.01:  # Avoid division by zero
                drop_percent = (prev_avg - curr_avg) / prev_avg * 100

                if drop_percent > 30:  # 30% speed drop threshold
                    events.append({
                        "timestamp": round(float(timestamps[i+1]), 2),
                        "type": "speed_drop",
                        "description": f"속도 급감 ({int(drop_percent)}% 감소)"
                    })
                    # Skip ahead to avoid duplicate detections
                    i += window

        return events[:5]  # Limit to 5 events

    def _detect_pauses(self, trajectories: Dict) -> List[Dict]:
        """Detect movement pauses (very low velocity for extended period)"""
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

        # Find baseline speed (median)
        baseline_speed = np.median(speeds)
        pause_threshold = baseline_speed * 0.1  # 10% of baseline = pause
        min_pause_frames = int(self.fps * 0.5)  # At least 0.5 seconds

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
                    events.append({
                        "timestamp": round(float(timestamps[pause_start+1]), 2),
                        "type": "pause",
                        "description": f"멈춤 감지 ({pause_duration:.1f}초)"
                    })
                pause_start = None
                pause_count = 0

        return events[:5]  # Limit to 5 events

    def _detect_fatigue(self, trajectories: Dict) -> List[Dict]:
        """Detect amplitude fatigue (progressive decrease in movement range)"""
        events = []

        if len(trajectories['timestamps']) < 60:  # Need enough data
            return events

        timestamps = trajectories['timestamps']

        # Use wrist movement for finger tapping
        left_wrist = trajectories['left_wrist']
        right_wrist = trajectories['right_wrist']

        # Calculate amplitude over time (sliding window)
        window_frames = int(self.fps * 1.0)  # 1 second window

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

                if fatigue_percent > 30:  # 30% amplitude decrease
                    events.append({
                        "timestamp": round(float(timestamps[len(timestamps)//2]), 2),
                        "type": "fatigue",
                        "description": f"피로 감지 - {side} 진폭 {int(fatigue_percent)}% 감소"
                    })

        return events

    def _detect_asymmetry_events(self, trajectories: Dict, gait_analysis: Optional[Dict]) -> List[Dict]:
        """Detect significant asymmetry events"""
        events = []

        if gait_analysis:
            asym = gait_analysis.get('asymmetry_percent', {})

            # Check step length asymmetry
            step_asym = abs(asym.get('step_length', 0))
            if step_asym > 15:  # >15% asymmetry is clinically significant
                events.append({
                    "timestamp": 0.0,
                    "type": "asymmetry",
                    "description": f"보폭 비대칭 {int(step_asym)}%"
                })

            # Check swing time asymmetry
            swing_asym = abs(asym.get('swing_time', 0))
            if swing_asym > 15:
                events.append({
                    "timestamp": 0.0,
                    "type": "asymmetry",
                    "description": f"스윙 시간 비대칭 {int(swing_asym)}%"
                })

        return events


def generate_visualization_data(landmark_frames: List[Dict],
                                gait_analysis: Optional[Dict] = None,
                                fps: float = 30.0) -> Dict:
    """
    Convenience function to generate visualization data

    Args:
        landmark_frames: List of frame data with landmarks
        gait_analysis: Optional pre-computed gait analysis
        fps: Video frame rate

    Returns:
        Dict with visualization data
    """
    generator = VisualizationDataGenerator(fps=fps)
    return generator.generate(landmark_frames, gait_analysis)


def detect_events(landmark_frames: List[Dict],
                  gait_analysis: Optional[Dict] = None,
                  fps: float = 30.0,
                  task_type: str = "gait") -> List[Dict]:
    """
    Convenience function to detect events

    Args:
        landmark_frames: List of frame data with landmarks
        gait_analysis: Optional pre-computed gait analysis
        fps: Video frame rate
        task_type: 'gait' or 'finger_tapping'

    Returns:
        List of detected events
    """
    generator = VisualizationDataGenerator(fps=fps)
    trajectories = generator._extract_trajectories(landmark_frames)

    detector = EventDetector(fps=fps)
    return detector.detect_events(trajectories, gait_analysis, task_type)
