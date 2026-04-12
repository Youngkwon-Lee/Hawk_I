"""
Gait Cycle Analyzer V2 - Improved Version
Enhanced gait event detection with multi-method approach

Improvements over V1:
1. Multi-method event detection (Zeni, Velocity, Height, Ensemble)
2. Automatic camera angle detection
3. Adaptive thresholding based on signal characteristics
4. Partial data handling with confidence scores
5. Robust preprocessing with outlier removal

Reference: Perry & Burnfield (2010) "Gait Analysis: Normal and Pathological Function"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.ndimage import median_filter
import json
import warnings


@dataclass
class GaitEvent:
    """Single gait event (heel strike or toe off)"""
    frame_idx: int
    timestamp: float
    event_type: str  # 'HS' (heel strike) or 'TO' (toe off)
    side: str  # 'L' (left) or 'R' (right)
    confidence: float = 1.0  # Detection confidence (0-1)
    detection_method: str = "ensemble"  # Which method detected this event


@dataclass
class GaitCycle:
    """Complete gait cycle from heel strike to next heel strike (same foot)"""
    cycle_number: int
    side: str

    # Events
    initial_contact_frame: int
    initial_contact_time: float
    toe_off_frame: int
    toe_off_time: float
    terminal_swing_frame: int
    terminal_swing_time: float

    # Phase durations (seconds)
    stance_time: float
    swing_time: float
    cycle_time: float

    # Phase percentages
    stance_percent: float
    swing_percent: float

    # Sub-phase timing (% of cycle)
    loading_response_percent: float
    mid_stance_percent: float
    terminal_stance_percent: float
    pre_swing_percent: float
    initial_swing_percent: float
    mid_swing_percent: float
    terminal_swing_percent: float

    # Spatial metrics
    step_length: float
    step_height: float
    arm_swing_amplitude: float

    # Joint angles
    hip_angle_at_ic: float
    knee_angle_at_ic: float
    ankle_angle_at_ic: float
    knee_angle_at_to: float
    peak_knee_flexion_swing: float

    # Quality metrics
    confidence: float = 1.0


@dataclass
class GaitCycleAnalysis:
    """Complete gait cycle analysis results"""
    # Summary
    num_cycles_left: int
    num_cycles_right: int
    total_cycles: int
    analysis_duration: float

    # Quality indicators
    overall_confidence: float
    detection_method_used: str
    camera_view: str  # 'sagittal', 'frontal', 'oblique', 'unknown'

    # Timing statistics
    cycle_time_mean: float
    cycle_time_std: float
    cycle_time_cv: float

    # Phase timing
    stance_percent_mean: float
    stance_percent_std: float
    swing_percent_mean: float
    swing_percent_std: float

    # Double/single support
    double_support_percent: float
    single_support_percent: float

    # Sub-phase variability (CV %)
    loading_response_cv: float
    mid_stance_cv: float
    terminal_stance_cv: float
    pre_swing_cv: float
    initial_swing_cv: float
    mid_swing_cv: float
    terminal_swing_cv: float

    # Asymmetry
    stance_time_asymmetry: float
    swing_time_asymmetry: float
    cycle_time_asymmetry: float
    step_length_asymmetry: float

    # Variability
    step_length_cv: float
    step_time_cv: float
    stride_time_cv: float

    # Individual cycles
    left_cycles: List[GaitCycle] = field(default_factory=list)
    right_cycles: List[GaitCycle] = field(default_factory=list)
    events: List[GaitEvent] = field(default_factory=list)

    # Partial analysis flag
    is_partial: bool = False
    partial_reason: str = ""


class GaitCycleAnalyzerV2:
    """
    Enhanced Gait Cycle Analyzer with multi-method detection
    """

    # Detection method weights for ensemble
    METHOD_WEIGHTS = {
        'zeni': 0.35,      # Position-based (Zeni method)
        'velocity': 0.35,  # Velocity-based
        'height': 0.20,    # Vertical position
        'angle': 0.10      # Joint angle-based
    }

    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    HIGH_CONFIDENCE = 0.7

    def __init__(self, fps: float = 30.0, verbose: bool = False):
        self.fps = fps
        self.dt = 1.0 / fps
        self.verbose = verbose

    def analyze(self, landmark_frames: List[Dict],
                min_events: int = 2,
                allow_partial: bool = True) -> GaitCycleAnalysis:
        """
        Perform gait cycle analysis with enhanced detection

        Args:
            landmark_frames: List of frames with pose landmarks
            min_events: Minimum events required (default 2 for partial analysis)
            allow_partial: Allow partial results if full analysis not possible

        Returns:
            GaitCycleAnalysis with detailed cycle breakdown
        """
        if len(landmark_frames) < 15:
            if allow_partial:
                return self._create_partial_analysis(
                    "insufficient_frames",
                    f"Only {len(landmark_frames)} frames (need 15+)"
                )
            raise ValueError(f"Insufficient data: {len(landmark_frames)} frames (need 15+)")

        # Extract trajectories
        trajectories = self._extract_trajectories(landmark_frames)

        if len(trajectories['timestamps']) < 10:
            if allow_partial:
                return self._create_partial_analysis(
                    "missing_landmarks",
                    "Too many frames missing required landmarks"
                )
            raise ValueError("Insufficient valid frames with landmarks")

        # Detect camera view angle
        camera_view = self._detect_camera_view(trajectories)
        if self.verbose:
            print(f"[GaitAnalyzer] Detected camera view: {camera_view}")

        # Multi-method event detection
        events, detection_method, confidence = self._detect_events_ensemble(
            trajectories, camera_view
        )

        if self.verbose:
            print(f"[GaitAnalyzer] Detected {len(events)} events using {detection_method}")
            print(f"[GaitAnalyzer] Detection confidence: {confidence:.2f}")

        # Check minimum events
        if len(events) < min_events:
            if allow_partial:
                return self._create_partial_analysis(
                    "insufficient_events",
                    f"Only {len(events)} gait events detected (need {min_events}+)",
                    events=events,
                    camera_view=camera_view
                )
            raise ValueError(f"Insufficient gait events: {len(events)} (need {min_events}+)")

        # Build gait cycles
        left_cycles = self._build_cycles(events, trajectories, side='L')
        right_cycles = self._build_cycles(events, trajectories, side='R')

        # Check if we have complete cycles
        all_cycles = left_cycles + right_cycles
        is_partial = len(all_cycles) == 0

        if is_partial:
            if allow_partial:
                return self._create_partial_analysis(
                    "no_complete_cycles",
                    f"Events detected but no complete gait cycles formed",
                    events=events,
                    camera_view=camera_view,
                    detection_method=detection_method,
                    confidence=confidence
                )
            raise ValueError("No complete gait cycles detected")

        # Calculate summary statistics
        analysis = self._calculate_summary(
            left_cycles, right_cycles, events, trajectories,
            camera_view, detection_method, confidence
        )

        return analysis

    def _extract_trajectories(self, landmark_frames: List[Dict]) -> Dict:
        """Extract and preprocess landmark trajectories"""
        trajectories = {
            'timestamps': [],
            'frame_indices': [],
            # Normalized landmarks
            'left_ankle': [], 'right_ankle': [],
            'left_hip': [], 'right_hip': [],
            'left_knee': [], 'right_knee': [],
            'left_shoulder': [], 'right_shoulder': [],
            'left_wrist': [], 'right_wrist': [],
            'left_toe': [], 'right_toe': [],
            'left_heel': [], 'right_heel': [],
            # World landmarks
            'left_ankle_world': [], 'right_ankle_world': [],
            'left_hip_world': [], 'right_hip_world': [],
            'left_knee_world': [], 'right_knee_world': [],
            'left_toe_world': [], 'right_toe_world': [],
            'left_heel_world': [], 'right_heel_world': [],
        }

        landmark_ids = {
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_wrist': 15, 'right_wrist': 16,
            'left_toe': 31, 'right_toe': 32,
            'left_heel': 29, 'right_heel': 30,
        }

        for frame_idx, frame in enumerate(landmark_frames):
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            kp_dict = {kp['id']: kp for kp in keypoints}

            world_kp = frame.get('world_keypoints', frame.get('world_landmarks', []))
            world_dict = {kp['id']: kp for kp in world_kp} if world_kp else {}

            # Check required landmarks (more lenient - need at least hips and one ankle)
            required = [23, 24]  # Only hips required
            ankle_present = 27 in kp_dict or 28 in kp_dict

            if not (all(k in kp_dict for k in required) and ankle_present):
                continue

            ts = frame.get('timestamp', frame.get('frame_number', frame_idx) * self.dt)
            trajectories['timestamps'].append(ts)
            trajectories['frame_indices'].append(frame_idx)

            # Extract landmarks with fallback
            for name, lid in landmark_ids.items():
                if lid in kp_dict:
                    trajectories[name].append([
                        kp_dict[lid]['x'],
                        kp_dict[lid]['y'],
                        kp_dict[lid].get('z', 0)
                    ])
                else:
                    # Interpolate or use last known value
                    if trajectories[name]:
                        trajectories[name].append(trajectories[name][-1])
                    else:
                        trajectories[name].append([0.5, 0.5, 0])

                # World landmarks
                world_key = f"{name}_world"
                if world_key in trajectories:
                    if lid in world_dict:
                        trajectories[world_key].append([
                            world_dict[lid]['x'],
                            world_dict[lid]['y'],
                            world_dict[lid].get('z', 0)
                        ])
                    elif trajectories[world_key]:
                        trajectories[world_key].append(trajectories[world_key][-1])

        # Convert to numpy arrays
        for key in trajectories:
            arr = np.array(trajectories[key])
            trajectories[key] = arr

        # Apply preprocessing
        trajectories = self._preprocess_trajectories(trajectories)

        return trajectories

    def _preprocess_trajectories(self, trajectories: Dict) -> Dict:
        """Apply noise filtering and outlier removal"""
        # List of position keys to filter
        position_keys = [
            'left_ankle', 'right_ankle', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_toe', 'right_toe',
            'left_heel', 'right_heel',
            'left_ankle_world', 'right_ankle_world',
            'left_hip_world', 'right_hip_world',
            'left_knee_world', 'right_knee_world',
            'left_toe_world', 'right_toe_world',
            'left_heel_world', 'right_heel_world',
        ]

        for key in position_keys:
            if key not in trajectories or len(trajectories[key]) < 5:
                continue

            arr = trajectories[key]
            if arr.ndim != 2:
                continue

            # Apply median filter to remove spikes
            for dim in range(arr.shape[1]):
                if len(arr[:, dim]) >= 3:
                    arr[:, dim] = median_filter(arr[:, dim], size=3)

            # Apply Savitzky-Golay filter for smoothing
            window = min(15, len(arr) // 2 * 2 - 1)  # Must be odd
            if window >= 5:
                for dim in range(arr.shape[1]):
                    try:
                        arr[:, dim] = savgol_filter(arr[:, dim], window, 3)
                    except:
                        pass

            trajectories[key] = arr

        return trajectories

    def _detect_camera_view(self, trajectories: Dict) -> str:
        """
        Detect camera viewing angle based on landmark movement patterns

        Returns:
            'sagittal' (side view), 'frontal' (front/back view),
            'oblique' (angled), or 'unknown'
        """
        if len(trajectories['timestamps']) < 10:
            return 'unknown'

        left_ankle = trajectories['left_ankle']
        right_ankle = trajectories['right_ankle']

        if len(left_ankle) < 10:
            return 'unknown'

        # Calculate movement variance in each axis
        def movement_variance(arr):
            if arr.ndim != 2 or arr.shape[1] < 2:
                return 0, 0, 0
            x_var = np.var(arr[:, 0])  # Horizontal
            y_var = np.var(arr[:, 1])  # Vertical
            z_var = np.var(arr[:, 2]) if arr.shape[1] > 2 else 0  # Depth
            return x_var, y_var, z_var

        l_x, l_y, l_z = movement_variance(left_ankle)
        r_x, r_y, r_z = movement_variance(right_ankle)

        avg_x = (l_x + r_x) / 2
        avg_y = (l_y + r_y) / 2
        avg_z = (l_z + r_z) / 2

        # Sagittal view: large X movement, small Y movement
        # Frontal view: small X movement, small Y movement, relies on Z
        # Oblique: mixed

        total_var = avg_x + avg_y + avg_z + 1e-8
        x_ratio = avg_x / total_var
        z_ratio = avg_z / total_var

        if x_ratio > 0.5:
            return 'sagittal'
        elif z_ratio > 0.4:
            return 'frontal'
        elif x_ratio > 0.3:
            return 'oblique'
        else:
            return 'unknown'

    def _detect_events_ensemble(self, trajectories: Dict,
                                 camera_view: str) -> Tuple[List[GaitEvent], str, float]:
        """
        Detect gait events using ensemble of multiple methods

        Returns:
            (events, detection_method, confidence)
        """
        timestamps = trajectories['timestamps']

        if len(timestamps) < 10:
            return [], 'none', 0.0

        # Try multiple detection methods
        methods_results = {}

        # Method 1: Zeni (position-based)
        try:
            zeni_events = self._detect_events_zeni(trajectories, camera_view)
            if len(zeni_events) >= 2:
                methods_results['zeni'] = zeni_events
        except Exception as e:
            if self.verbose:
                print(f"[GaitAnalyzer] Zeni method failed: {e}")

        # Method 2: Velocity-based
        try:
            velocity_events = self._detect_events_velocity(trajectories)
            if len(velocity_events) >= 2:
                methods_results['velocity'] = velocity_events
        except Exception as e:
            if self.verbose:
                print(f"[GaitAnalyzer] Velocity method failed: {e}")

        # Method 3: Height-based (Y-coordinate)
        try:
            height_events = self._detect_events_height(trajectories)
            if len(height_events) >= 2:
                methods_results['height'] = height_events
        except Exception as e:
            if self.verbose:
                print(f"[GaitAnalyzer] Height method failed: {e}")

        # Method 4: Ankle angle-based (simplified)
        try:
            angle_events = self._detect_events_angle(trajectories)
            if len(angle_events) >= 2:
                methods_results['angle'] = angle_events
        except Exception as e:
            if self.verbose:
                print(f"[GaitAnalyzer] Angle method failed: {e}")

        if not methods_results:
            return [], 'none', 0.0

        # If only one method worked, use it
        if len(methods_results) == 1:
            method_name = list(methods_results.keys())[0]
            events = methods_results[method_name]
            confidence = 0.5  # Single method = moderate confidence
            return events, method_name, confidence

        # Ensemble: combine results from multiple methods
        events, confidence = self._combine_detection_results(methods_results, timestamps)

        return events, 'ensemble', confidence

    def _detect_events_zeni(self, trajectories: Dict,
                            camera_view: str) -> List[GaitEvent]:
        """
        Zeni method: Detect events based on foot position relative to hip
        """
        events = []
        timestamps = trajectories['timestamps']

        # Calculate hip center
        hip_center = (trajectories['left_hip'] + trajectories['right_hip']) / 2

        # Choose coordinate based on camera view
        if camera_view in ['sagittal', 'oblique']:
            # Side view: use X (horizontal) coordinate
            coord_idx = 0
        else:
            # Frontal view: use Z (depth) coordinate
            coord_idx = 2

        # Relative foot position
        left_ankle = trajectories['left_ankle']
        right_ankle = trajectories['right_ankle']

        left_rel = left_ankle[:, coord_idx] - hip_center[:, coord_idx]
        right_rel = right_ankle[:, coord_idx] - hip_center[:, coord_idx]

        # Adaptive thresholding
        left_range = np.ptp(left_rel)
        right_range = np.ptp(right_rel)

        # Dynamic prominence based on signal range
        left_prominence = max(0.01, 0.015 * left_range)
        right_prominence = max(0.01, 0.015 * right_range)

        # Dynamic minimum distance based on expected cadence
        # Normal cadence: 90-130 steps/min = 0.46-0.67 sec per step
        # PD cadence: can be 60-180 steps/min = 0.33-1.0 sec per step
        min_step_time = 0.25  # 240 steps/min max
        max_step_time = 1.5   # 40 steps/min min
        min_distance = int(min_step_time * self.fps)

        # Detect heel strikes (foot furthest forward = local max or min depending on direction)
        # Try both polarities and use the one that gives more consistent results

        for polarity in [1, -1]:
            signal = polarity * left_rel
            peaks, props = find_peaks(signal,
                                     distance=min_distance,
                                     prominence=left_prominence)
            if len(peaks) >= 2:
                for idx in peaks:
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='HS',
                        side='L',
                        confidence=min(1.0, props['prominences'][list(peaks).index(idx)] / left_range) if left_range > 0 else 0.5,
                        detection_method='zeni'
                    ))
                break

        for polarity in [1, -1]:
            signal = polarity * right_rel
            peaks, props = find_peaks(signal,
                                     distance=min_distance,
                                     prominence=right_prominence)
            if len(peaks) >= 2:
                for idx in peaks:
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='HS',
                        side='R',
                        confidence=min(1.0, props['prominences'][list(peaks).index(idx)] / right_range) if right_range > 0 else 0.5,
                        detection_method='zeni'
                    ))
                break

        # Detect toe-offs (foot furthest backward = local min)
        for side, rel_signal, range_val in [('L', left_rel, left_range), ('R', right_rel, right_range)]:
            prominence = max(0.01, 0.015 * range_val)

            for polarity in [1, -1]:
                signal = polarity * (-rel_signal)  # Inverted for toe-off
                peaks, props = find_peaks(signal,
                                         distance=min_distance,
                                         prominence=prominence)
                if len(peaks) >= 1:
                    for idx in peaks:
                        events.append(GaitEvent(
                            frame_idx=int(idx),
                            timestamp=float(timestamps[idx]),
                            event_type='TO',
                            side=side,
                            confidence=min(1.0, props['prominences'][list(peaks).index(idx)] / range_val) if range_val > 0 else 0.5,
                            detection_method='zeni'
                        ))
                    break

        events.sort(key=lambda e: e.timestamp)
        return events

    def _detect_events_velocity(self, trajectories: Dict) -> List[GaitEvent]:
        """
        Velocity-based detection: Heel strike when foot velocity approaches zero
        """
        events = []
        timestamps = trajectories['timestamps']

        if len(timestamps) < 5:
            return events

        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            dt = self.dt

        for side in ['L', 'R']:
            ankle_key = 'left_ankle' if side == 'L' else 'right_ankle'
            ankle = trajectories[ankle_key]

            if len(ankle) < 5:
                continue

            # Calculate velocity magnitude
            velocity = np.diff(ankle, axis=0) / dt
            velocity_mag = np.linalg.norm(velocity, axis=1)

            # Smooth velocity
            if len(velocity_mag) >= 5:
                velocity_mag = savgol_filter(velocity_mag, min(11, len(velocity_mag)//2*2+1), 3)

            # Heel strike: local minimum in velocity (foot decelerating to contact)
            # Pad with edge values
            velocity_mag = np.concatenate([[velocity_mag[0]], velocity_mag])

            # Find local minima (valleys)
            inverted = -velocity_mag
            min_distance = int(0.25 * self.fps)
            peaks, props = find_peaks(inverted,
                                     distance=min_distance,
                                     prominence=0.05 * np.ptp(velocity_mag))

            for idx in peaks:
                if idx < len(timestamps):
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='HS',
                        side=side,
                        confidence=0.6,
                        detection_method='velocity'
                    ))

            # Toe-off: local maximum in velocity (foot accelerating into swing)
            peaks_to, _ = find_peaks(velocity_mag,
                                    distance=min_distance,
                                    prominence=0.05 * np.ptp(velocity_mag))

            for idx in peaks_to:
                if idx < len(timestamps):
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='TO',
                        side=side,
                        confidence=0.6,
                        detection_method='velocity'
                    ))

        events.sort(key=lambda e: e.timestamp)
        return events

    def _detect_events_height(self, trajectories: Dict) -> List[GaitEvent]:
        """
        Height-based detection: Use vertical (Y) position of ankle/heel
        Heel strike when foot is at lowest point, toe-off when lifting
        """
        events = []
        timestamps = trajectories['timestamps']

        if len(timestamps) < 5:
            return events

        for side in ['L', 'R']:
            # Try heel first, then ankle
            for landmark in ['heel', 'ankle']:
                key = f"left_{landmark}" if side == 'L' else f"right_{landmark}"

                if key not in trajectories or len(trajectories[key]) < 5:
                    continue

                pos = trajectories[key]
                y_pos = pos[:, 1]  # Y coordinate (vertical in normalized coords, down is positive)

                # Smooth
                if len(y_pos) >= 5:
                    y_pos = savgol_filter(y_pos, min(11, len(y_pos)//2*2+1), 3)

                # Heel strike: local maximum in Y (foot at lowest point, Y increases downward)
                min_distance = int(0.25 * self.fps)
                peaks_hs, props = find_peaks(y_pos,
                                            distance=min_distance,
                                            prominence=0.01 * np.ptp(y_pos))

                for idx in peaks_hs:
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='HS',
                        side=side,
                        confidence=0.5,
                        detection_method='height'
                    ))

                # Toe-off: local minimum in Y (foot at highest point before swing)
                peaks_to, _ = find_peaks(-y_pos,
                                        distance=min_distance,
                                        prominence=0.01 * np.ptp(y_pos))

                for idx in peaks_to:
                    events.append(GaitEvent(
                        frame_idx=int(idx),
                        timestamp=float(timestamps[idx]),
                        event_type='TO',
                        side=side,
                        confidence=0.5,
                        detection_method='height'
                    ))

                if peaks_hs.size > 0 or peaks_to.size > 0:
                    break  # Found events with this landmark

        events.sort(key=lambda e: e.timestamp)
        return events

    def _detect_events_angle(self, trajectories: Dict) -> List[GaitEvent]:
        """
        Angle-based detection: Use knee angle changes
        """
        events = []
        timestamps = trajectories['timestamps']

        if len(timestamps) < 10:
            return events

        for side in ['L', 'R']:
            prefix = 'left' if side == 'L' else 'right'

            hip = trajectories[f'{prefix}_hip']
            knee = trajectories[f'{prefix}_knee']
            ankle = trajectories[f'{prefix}_ankle']

            if len(hip) < 10:
                continue

            # Calculate knee angle over time
            knee_angles = []
            for i in range(len(hip)):
                thigh = knee[i] - hip[i]
                shank = ankle[i] - knee[i]

                cos_angle = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                knee_angles.append(angle)

            knee_angles = np.array(knee_angles)

            # Smooth
            if len(knee_angles) >= 5:
                knee_angles = savgol_filter(knee_angles, min(11, len(knee_angles)//2*2+1), 3)

            # Heel strike: knee angle is at local minimum (more extended)
            min_distance = int(0.3 * self.fps)
            peaks_hs, _ = find_peaks(-knee_angles,  # Inverted for minimum
                                    distance=min_distance,
                                    prominence=5)  # 5 degrees prominence

            for idx in peaks_hs:
                events.append(GaitEvent(
                    frame_idx=int(idx),
                    timestamp=float(timestamps[idx]),
                    event_type='HS',
                    side=side,
                    confidence=0.4,
                    detection_method='angle'
                ))

            # Toe-off: knee angle at intermediate value, velocity changing sign
            # This is harder to detect reliably, so we skip it here

        events.sort(key=lambda e: e.timestamp)
        return events

    def _combine_detection_results(self, methods_results: Dict[str, List[GaitEvent]],
                                   timestamps: np.ndarray) -> Tuple[List[GaitEvent], float]:
        """
        Combine events from multiple detection methods using clustering
        """
        if not methods_results:
            return [], 0.0

        # Time window for considering events as the same (in seconds)
        merge_window = 0.1  # 100ms

        all_events = []
        for method, events in methods_results.items():
            weight = self.METHOD_WEIGHTS.get(method, 0.25)
            for event in events:
                all_events.append({
                    'event': event,
                    'weight': weight,
                    'method': method
                })

        if not all_events:
            return [], 0.0

        # Sort by timestamp
        all_events.sort(key=lambda x: x['event'].timestamp)

        # Cluster events that are close in time
        clusters = []
        current_cluster = [all_events[0]]

        for ev in all_events[1:]:
            if ev['event'].timestamp - current_cluster[-1]['event'].timestamp <= merge_window:
                # Same cluster if same type and side
                if (ev['event'].event_type == current_cluster[0]['event'].event_type and
                    ev['event'].side == current_cluster[0]['event'].side):
                    current_cluster.append(ev)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [ev]
            else:
                clusters.append(current_cluster)
                current_cluster = [ev]

        clusters.append(current_cluster)

        # Create merged events
        merged_events = []
        total_confidence = 0

        for cluster in clusters:
            if not cluster:
                continue

            # Weighted average timestamp
            total_weight = sum(c['weight'] for c in cluster)
            avg_timestamp = sum(c['event'].timestamp * c['weight'] for c in cluster) / total_weight

            # Find closest frame
            frame_idx = int(np.argmin(np.abs(timestamps - avg_timestamp)))

            # Confidence based on number of methods that agree
            methods_agreed = len(set(c['method'] for c in cluster))
            confidence = min(1.0, 0.3 + 0.2 * methods_agreed + 0.1 * len(cluster))
            total_confidence += confidence

            merged_event = GaitEvent(
                frame_idx=frame_idx,
                timestamp=float(timestamps[frame_idx]) if frame_idx < len(timestamps) else avg_timestamp,
                event_type=cluster[0]['event'].event_type,
                side=cluster[0]['event'].side,
                confidence=confidence,
                detection_method='ensemble'
            )

            merged_events.append(merged_event)

        # Sort final events
        merged_events.sort(key=lambda e: e.timestamp)

        # Calculate overall confidence
        overall_confidence = total_confidence / len(merged_events) if merged_events else 0.0
        overall_confidence *= (len(methods_results) / 4)  # Scale by number of methods that worked
        overall_confidence = min(1.0, overall_confidence)

        return merged_events, overall_confidence

    def _build_cycles(self, events: List[GaitEvent], trajectories: Dict,
                      side: str) -> List[GaitCycle]:
        """Build complete gait cycles for one side"""
        cycles = []

        hs_events = [e for e in events if e.side == side and e.event_type == 'HS']
        to_events = [e for e in events if e.side == side and e.event_type == 'TO']

        if len(hs_events) < 2:
            return cycles

        for i in range(len(hs_events) - 1):
            ic_event = hs_events[i]
            next_ic_event = hs_events[i + 1]

            # Find toe-off between these heel strikes
            to_between = [e for e in to_events
                         if ic_event.timestamp < e.timestamp < next_ic_event.timestamp]

            if not to_between:
                # Estimate toe-off at ~60% of cycle
                estimated_to_time = ic_event.timestamp + 0.6 * (next_ic_event.timestamp - ic_event.timestamp)
                to_event = GaitEvent(
                    frame_idx=int(ic_event.frame_idx + 0.6 * (next_ic_event.frame_idx - ic_event.frame_idx)),
                    timestamp=estimated_to_time,
                    event_type='TO',
                    side=side,
                    confidence=0.3,
                    detection_method='estimated'
                )
            else:
                to_event = to_between[0]

            # Calculate timing
            stance_time = to_event.timestamp - ic_event.timestamp
            swing_time = next_ic_event.timestamp - to_event.timestamp
            cycle_time = next_ic_event.timestamp - ic_event.timestamp

            if cycle_time <= 0.2 or cycle_time > 3.0:  # Invalid cycle time
                continue

            stance_percent = (stance_time / cycle_time) * 100
            swing_percent = (swing_time / cycle_time) * 100

            # Validate phase percentages (stance typically 55-65%, swing 35-45%)
            if stance_percent < 30 or stance_percent > 85:
                # Abnormal but might be valid for PD patients
                pass

            # Sub-phase percentages (based on Perry & Burnfield)
            loading_response_percent = min(10.0, stance_percent * 0.17)
            mid_stance_percent = stance_percent * 0.33
            terminal_stance_percent = stance_percent * 0.33
            pre_swing_percent = stance_percent * 0.17
            initial_swing_percent = swing_percent * 0.33
            mid_swing_percent = swing_percent * 0.34
            terminal_swing_percent = swing_percent * 0.33

            # Spatial metrics
            step_length = self._calculate_step_length(trajectories, ic_event.frame_idx, next_ic_event.frame_idx, side)
            step_height = self._calculate_step_height(trajectories, to_event.frame_idx, next_ic_event.frame_idx, side)
            arm_swing = self._calculate_arm_swing(trajectories, ic_event.frame_idx, next_ic_event.frame_idx, side)

            # Joint angles
            hip_ic, knee_ic, ankle_ic = self._get_joint_angles_at_frame(trajectories, ic_event.frame_idx, side)
            _, knee_to, _ = self._get_joint_angles_at_frame(trajectories, to_event.frame_idx, side)
            peak_knee_swing = self._get_peak_knee_flexion(trajectories, to_event.frame_idx, next_ic_event.frame_idx, side)

            # Cycle confidence
            cycle_confidence = (ic_event.confidence + to_event.confidence + next_ic_event.confidence) / 3

            cycle = GaitCycle(
                cycle_number=i + 1,
                side=side,
                initial_contact_frame=ic_event.frame_idx,
                initial_contact_time=ic_event.timestamp,
                toe_off_frame=to_event.frame_idx,
                toe_off_time=to_event.timestamp,
                terminal_swing_frame=next_ic_event.frame_idx,
                terminal_swing_time=next_ic_event.timestamp,
                stance_time=float(stance_time),
                swing_time=float(swing_time),
                cycle_time=float(cycle_time),
                stance_percent=float(stance_percent),
                swing_percent=float(swing_percent),
                loading_response_percent=float(loading_response_percent),
                mid_stance_percent=float(mid_stance_percent),
                terminal_stance_percent=float(terminal_stance_percent),
                pre_swing_percent=float(pre_swing_percent),
                initial_swing_percent=float(initial_swing_percent),
                mid_swing_percent=float(mid_swing_percent),
                terminal_swing_percent=float(terminal_swing_percent),
                step_length=float(step_length),
                step_height=float(step_height),
                arm_swing_amplitude=float(arm_swing),
                hip_angle_at_ic=float(hip_ic),
                knee_angle_at_ic=float(knee_ic),
                ankle_angle_at_ic=float(ankle_ic),
                knee_angle_at_to=float(knee_to),
                peak_knee_flexion_swing=float(peak_knee_swing),
                confidence=float(cycle_confidence)
            )

            cycles.append(cycle)

        return cycles

    def _calculate_step_length(self, trajectories: Dict, start_idx: int,
                               end_idx: int, side: str) -> float:
        """Calculate step length with better scaling"""
        ankle_key = 'left_ankle' if side == 'L' else 'right_ankle'

        ankle = trajectories.get(ankle_key, [])
        if start_idx >= len(ankle) or end_idx >= len(ankle):
            return 0.0

        start_pos = ankle[start_idx]
        end_pos = ankle[end_idx]

        dist_norm = np.linalg.norm(end_pos - start_pos)

        # Scale using hip width
        left_hip = trajectories['left_hip']
        right_hip = trajectories['right_hip']
        hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
        valid_widths = hip_widths[hip_widths > 0.01]

        if len(valid_widths) == 0:
            return 0.0

        avg_hip_width = np.mean(valid_widths)
        REAL_HIP_WIDTH = 0.30  # meters
        scale = REAL_HIP_WIDTH / avg_hip_width if avg_hip_width > 0 else 2.0
        scale = np.clip(scale, 1.0, 15.0)

        step_length = dist_norm * scale
        return step_length if 0.05 < step_length < 2.5 else 0.0

    def _calculate_step_height(self, trajectories: Dict, to_idx: int,
                               hs_idx: int, side: str) -> float:
        """Calculate foot clearance during swing"""
        # Try world landmarks first
        world_key = 'left_ankle_world' if side == 'L' else 'right_ankle_world'
        norm_key = 'left_ankle' if side == 'L' else 'right_ankle'

        ankle = trajectories.get(world_key, trajectories.get(norm_key, []))

        if to_idx >= len(ankle) or hs_idx >= len(ankle) or hs_idx <= to_idx:
            return 0.0

        swing_y = ankle[to_idx:hs_idx, 1]
        if len(swing_y) < 2:
            return 0.0

        baseline_y = min(ankle[to_idx, 1], ankle[min(hs_idx, len(ankle)-1), 1])
        peak_y = np.min(swing_y)  # Y decreases when going up in normalized coords
        step_height = abs(baseline_y - peak_y)

        return step_height if 0.005 < step_height < 0.5 else 0.0

    def _calculate_arm_swing(self, trajectories: Dict, start_idx: int,
                             end_idx: int, side: str) -> float:
        """Calculate contralateral arm swing"""
        wrist_key = 'right_wrist' if side == 'L' else 'left_wrist'

        wrist = trajectories.get(wrist_key, [])
        if start_idx >= len(wrist) or end_idx >= len(wrist):
            return 0.0

        segment = wrist[start_idx:end_idx]
        if len(segment) < 3:
            return 0.0

        amplitude_norm = np.ptp(segment[:, 0])

        # Scale
        left_hip = trajectories['left_hip']
        right_hip = trajectories['right_hip']
        hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
        valid_widths = hip_widths[hip_widths > 0.01]

        if len(valid_widths) == 0:
            return amplitude_norm * 2.0

        avg_hip_width = np.mean(valid_widths)
        REAL_HIP_WIDTH = 0.30
        scale = REAL_HIP_WIDTH / avg_hip_width if avg_hip_width > 0 else 2.0

        return amplitude_norm * scale

    def _get_joint_angles_at_frame(self, trajectories: Dict, frame_idx: int,
                                   side: str) -> Tuple[float, float, float]:
        """Get hip, knee, ankle angles at frame"""
        prefix = 'left' if side == 'L' else 'right'

        # Try world then normalized
        hip = self._get_landmark_at_frame(trajectories, f"{prefix}_hip", frame_idx)
        knee = self._get_landmark_at_frame(trajectories, f"{prefix}_knee", frame_idx)
        ankle = self._get_landmark_at_frame(trajectories, f"{prefix}_ankle", frame_idx)

        if hip is None or knee is None or ankle is None:
            return 0.0, 0.0, 0.0

        # Hip flexion
        thigh_vec = knee - hip
        vertical = np.array([0, 1, 0])
        hip_angle = self._angle_between(thigh_vec, vertical)

        # Knee flexion
        shank_vec = ankle - knee
        knee_angle = 180 - self._angle_between(-thigh_vec, shank_vec)

        return hip_angle, knee_angle, 90.0

    def _get_landmark_at_frame(self, trajectories: Dict, key: str,
                               frame_idx: int) -> Optional[np.ndarray]:
        """Safely get landmark at frame index"""
        # Try world first
        world_key = f"{key}_world"
        for k in [world_key, key]:
            if k in trajectories and len(trajectories[k]) > frame_idx:
                return np.array(trajectories[k][frame_idx])
        return None

    def _get_peak_knee_flexion(self, trajectories: Dict, to_idx: int,
                               hs_idx: int, side: str) -> float:
        """Get peak knee flexion during swing"""
        prefix = 'left' if side == 'L' else 'right'

        hip = trajectories.get(f"{prefix}_hip", [])
        knee = trajectories.get(f"{prefix}_knee", [])
        ankle = trajectories.get(f"{prefix}_ankle", [])

        if len(hip) <= hs_idx or len(knee) <= hs_idx or len(ankle) <= hs_idx:
            return 0.0

        peak_flexion = 0.0
        for i in range(to_idx, min(hs_idx, len(hip))):
            h = np.array(hip[i])
            k = np.array(knee[i])
            a = np.array(ankle[i])

            thigh = k - h
            shank = a - k
            angle = 180 - self._angle_between(-thigh, shank)

            if angle > peak_flexion:
                peak_flexion = angle

        return peak_flexion

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle between vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _calculate_summary(self, left_cycles: List[GaitCycle],
                          right_cycles: List[GaitCycle],
                          events: List[GaitEvent],
                          trajectories: Dict,
                          camera_view: str,
                          detection_method: str,
                          detection_confidence: float) -> GaitCycleAnalysis:
        """Calculate summary statistics"""
        all_cycles = left_cycles + right_cycles

        timestamps = trajectories['timestamps']
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0

        # Cycle timing
        cycle_times = [c.cycle_time for c in all_cycles]
        cycle_time_mean = float(np.mean(cycle_times)) if cycle_times else 0
        cycle_time_std = float(np.std(cycle_times)) if cycle_times else 0
        cycle_time_cv = (cycle_time_std / cycle_time_mean * 100) if cycle_time_mean > 0 else 0

        # Phase timing
        stance_percents = [c.stance_percent for c in all_cycles]
        swing_percents = [c.swing_percent for c in all_cycles]

        stance_percent_mean = float(np.mean(stance_percents)) if stance_percents else 60.0
        stance_percent_std = float(np.std(stance_percents)) if stance_percents else 0
        swing_percent_mean = float(np.mean(swing_percents)) if swing_percents else 40.0
        swing_percent_std = float(np.std(swing_percents)) if swing_percents else 0

        # Double support
        double_support_percent = max(0, 100 - swing_percent_mean * 2)
        single_support_percent = 100 - double_support_percent

        # Sub-phase CV
        def calc_cv(values):
            if len(values) < 2:
                return 0.0
            mean_val = np.mean(values)
            return (np.std(values) / mean_val * 100) if mean_val > 0 else 0.0

        loading_response_cv = calc_cv([c.loading_response_percent for c in all_cycles])
        mid_stance_cv = calc_cv([c.mid_stance_percent for c in all_cycles])
        terminal_stance_cv = calc_cv([c.terminal_stance_percent for c in all_cycles])
        pre_swing_cv = calc_cv([c.pre_swing_percent for c in all_cycles])
        initial_swing_cv = calc_cv([c.initial_swing_percent for c in all_cycles])
        mid_swing_cv = calc_cv([c.mid_swing_percent for c in all_cycles])
        terminal_swing_cv = calc_cv([c.terminal_swing_percent for c in all_cycles])

        # Asymmetry
        def calc_asymmetry(left_vals, right_vals):
            if not left_vals or not right_vals:
                return 0.0
            left_mean = np.mean(left_vals)
            right_mean = np.mean(right_vals)
            avg = (left_mean + right_mean) / 2
            return abs(left_mean - right_mean) / avg * 100 if avg > 0 else 0.0

        stance_time_asymmetry = calc_asymmetry(
            [c.stance_time for c in left_cycles],
            [c.stance_time for c in right_cycles])
        swing_time_asymmetry = calc_asymmetry(
            [c.swing_time for c in left_cycles],
            [c.swing_time for c in right_cycles])
        cycle_time_asymmetry = calc_asymmetry(
            [c.cycle_time for c in left_cycles],
            [c.cycle_time for c in right_cycles])
        step_length_asymmetry = calc_asymmetry(
            [c.step_length for c in left_cycles if c.step_length > 0],
            [c.step_length for c in right_cycles if c.step_length > 0])

        # Variability
        step_lengths = [c.step_length for c in all_cycles if c.step_length > 0]
        step_length_cv = calc_cv(step_lengths)
        step_time_cv = calc_cv([c.stance_time for c in all_cycles])
        stride_time_cv = cycle_time_cv

        # Overall confidence
        cycle_confidences = [c.confidence for c in all_cycles]
        overall_confidence = float(np.mean(cycle_confidences)) if cycle_confidences else detection_confidence

        return GaitCycleAnalysis(
            num_cycles_left=len(left_cycles),
            num_cycles_right=len(right_cycles),
            total_cycles=len(all_cycles),
            analysis_duration=float(duration),
            overall_confidence=overall_confidence,
            detection_method_used=detection_method,
            camera_view=camera_view,
            cycle_time_mean=cycle_time_mean,
            cycle_time_std=cycle_time_std,
            cycle_time_cv=float(cycle_time_cv),
            stance_percent_mean=float(stance_percent_mean),
            stance_percent_std=float(stance_percent_std),
            swing_percent_mean=float(swing_percent_mean),
            swing_percent_std=float(swing_percent_std),
            double_support_percent=float(double_support_percent),
            single_support_percent=float(single_support_percent),
            loading_response_cv=float(loading_response_cv),
            mid_stance_cv=float(mid_stance_cv),
            terminal_stance_cv=float(terminal_stance_cv),
            pre_swing_cv=float(pre_swing_cv),
            initial_swing_cv=float(initial_swing_cv),
            mid_swing_cv=float(mid_swing_cv),
            terminal_swing_cv=float(terminal_swing_cv),
            stance_time_asymmetry=float(stance_time_asymmetry),
            swing_time_asymmetry=float(swing_time_asymmetry),
            cycle_time_asymmetry=float(cycle_time_asymmetry),
            step_length_asymmetry=float(step_length_asymmetry),
            step_length_cv=float(step_length_cv),
            step_time_cv=float(step_time_cv),
            stride_time_cv=float(stride_time_cv),
            left_cycles=left_cycles,
            right_cycles=right_cycles,
            events=events,
            is_partial=False,
            partial_reason=""
        )

    def _create_partial_analysis(self, reason: str, message: str,
                                 events: List[GaitEvent] = None,
                                 camera_view: str = "unknown",
                                 detection_method: str = "none",
                                 confidence: float = 0.0) -> GaitCycleAnalysis:
        """Create a partial analysis result when full analysis not possible"""
        return GaitCycleAnalysis(
            num_cycles_left=0,
            num_cycles_right=0,
            total_cycles=0,
            analysis_duration=0.0,
            overall_confidence=confidence,
            detection_method_used=detection_method,
            camera_view=camera_view,
            cycle_time_mean=0.0,
            cycle_time_std=0.0,
            cycle_time_cv=0.0,
            stance_percent_mean=60.0,  # Default normal values
            stance_percent_std=0.0,
            swing_percent_mean=40.0,
            swing_percent_std=0.0,
            double_support_percent=20.0,
            single_support_percent=80.0,
            loading_response_cv=0.0,
            mid_stance_cv=0.0,
            terminal_stance_cv=0.0,
            pre_swing_cv=0.0,
            initial_swing_cv=0.0,
            mid_swing_cv=0.0,
            terminal_swing_cv=0.0,
            stance_time_asymmetry=0.0,
            swing_time_asymmetry=0.0,
            cycle_time_asymmetry=0.0,
            step_length_asymmetry=0.0,
            step_length_cv=0.0,
            step_time_cv=0.0,
            stride_time_cv=0.0,
            left_cycles=[],
            right_cycles=[],
            events=events or [],
            is_partial=True,
            partial_reason=message
        )

    def to_dict(self, analysis: GaitCycleAnalysis) -> Dict:
        """Convert analysis to dictionary for JSON serialization"""
        return {
            'summary': {
                'num_cycles_left': analysis.num_cycles_left,
                'num_cycles_right': analysis.num_cycles_right,
                'total_cycles': analysis.total_cycles,
                'analysis_duration_sec': analysis.analysis_duration,
                'overall_confidence': analysis.overall_confidence,
                'detection_method': analysis.detection_method_used,
                'camera_view': analysis.camera_view,
                'is_partial': analysis.is_partial,
                'partial_reason': analysis.partial_reason,
            },
            'timing': {
                'cycle_time_mean_sec': analysis.cycle_time_mean,
                'cycle_time_std_sec': analysis.cycle_time_std,
                'cycle_time_cv_percent': analysis.cycle_time_cv,
            },
            'phase_distribution': {
                'stance_percent_mean': analysis.stance_percent_mean,
                'stance_percent_std': analysis.stance_percent_std,
                'swing_percent_mean': analysis.swing_percent_mean,
                'swing_percent_std': analysis.swing_percent_std,
                'double_support_percent': analysis.double_support_percent,
                'single_support_percent': analysis.single_support_percent,
            },
            'sub_phase_variability_cv': {
                'loading_response': analysis.loading_response_cv,
                'mid_stance': analysis.mid_stance_cv,
                'terminal_stance': analysis.terminal_stance_cv,
                'pre_swing': analysis.pre_swing_cv,
                'initial_swing': analysis.initial_swing_cv,
                'mid_swing': analysis.mid_swing_cv,
                'terminal_swing': analysis.terminal_swing_cv,
            },
            'asymmetry_percent': {
                'stance_time': analysis.stance_time_asymmetry,
                'swing_time': analysis.swing_time_asymmetry,
                'cycle_time': analysis.cycle_time_asymmetry,
                'step_length': analysis.step_length_asymmetry,
            },
            'variability_cv': {
                'step_length': analysis.step_length_cv,
                'step_time': analysis.step_time_cv,
                'stride_time': analysis.stride_time_cv,
            },
            'cycles': {
                'left': [self._cycle_to_dict(c) for c in analysis.left_cycles],
                'right': [self._cycle_to_dict(c) for c in analysis.right_cycles],
            },
            'events': [
                {
                    'frame': e.frame_idx,
                    'time': e.timestamp,
                    'type': e.event_type,
                    'side': e.side,
                    'confidence': e.confidence,
                    'method': e.detection_method
                }
                for e in analysis.events
            ]
        }

    def _cycle_to_dict(self, cycle: GaitCycle) -> Dict:
        """Convert single cycle to dictionary"""
        return {
            'cycle_number': cycle.cycle_number,
            'confidence': cycle.confidence,
            'timing': {
                'ic_frame': cycle.initial_contact_frame,
                'ic_time': cycle.initial_contact_time,
                'to_frame': cycle.toe_off_frame,
                'to_time': cycle.toe_off_time,
                'next_ic_frame': cycle.terminal_swing_frame,
                'next_ic_time': cycle.terminal_swing_time,
            },
            'durations_sec': {
                'stance': cycle.stance_time,
                'swing': cycle.swing_time,
                'total': cycle.cycle_time,
            },
            'phase_percent': {
                'stance': cycle.stance_percent,
                'swing': cycle.swing_percent,
            },
            'sub_phase_percent': {
                'loading_response': cycle.loading_response_percent,
                'mid_stance': cycle.mid_stance_percent,
                'terminal_stance': cycle.terminal_stance_percent,
                'pre_swing': cycle.pre_swing_percent,
                'initial_swing': cycle.initial_swing_percent,
                'mid_swing': cycle.mid_swing_percent,
                'terminal_swing': cycle.terminal_swing_percent,
            },
            'spatial_meters': {
                'step_length': cycle.step_length,
                'step_height': cycle.step_height,
                'arm_swing': cycle.arm_swing_amplitude,
            },
            'joint_angles_deg': {
                'hip_at_ic': cycle.hip_angle_at_ic,
                'knee_at_ic': cycle.knee_angle_at_ic,
                'ankle_at_ic': cycle.ankle_angle_at_ic,
                'knee_at_to': cycle.knee_angle_at_to,
                'peak_knee_swing': cycle.peak_knee_flexion_swing,
            }
        }


def analyze_gait_cycles_v2(landmark_frames: List[Dict],
                           fps: float = 30.0,
                           allow_partial: bool = True,
                           verbose: bool = False) -> Dict:
    """
    Convenience function for gait cycle analysis

    Args:
        landmark_frames: List of frames with pose landmarks
        fps: Video frame rate
        allow_partial: Allow partial results
        verbose: Print debug info

    Returns:
        Dictionary with gait cycle analysis results
    """
    analyzer = GaitCycleAnalyzerV2(fps=fps, verbose=verbose)
    analysis = analyzer.analyze(landmark_frames, allow_partial=allow_partial)
    return analyzer.to_dict(analysis)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gait_cycle_analyzer_v2.py <skeleton_json> [fps]")
        sys.exit(1)

    skeleton_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    print(f"[GaitCycleAnalyzer V2] Analyzing: {skeleton_path}")
    print(f"[GaitCycleAnalyzer V2] FPS: {fps}")

    with open(skeleton_path, 'r') as f:
        landmark_frames = json.load(f)

    result = analyze_gait_cycles_v2(landmark_frames, fps=fps, verbose=True)

    print("\n" + "="*60)
    print("GAIT CYCLE ANALYSIS RESULTS (V2)")
    print("="*60)

    summary = result['summary']
    print(f"\nTotal Cycles: {summary['total_cycles']} (L:{summary['num_cycles_left']}, R:{summary['num_cycles_right']})")
    print(f"Duration: {summary['analysis_duration_sec']:.1f} sec")
    print(f"Confidence: {summary['overall_confidence']:.2f}")
    print(f"Detection Method: {summary['detection_method']}")
    print(f"Camera View: {summary['camera_view']}")

    if summary['is_partial']:
        print(f"\n[PARTIAL ANALYSIS] {summary['partial_reason']}")
    else:
        timing = result['timing']
        print(f"\nCycle Time: {timing['cycle_time_mean_sec']:.3f} ± {timing['cycle_time_std_sec']:.3f} sec")
        print(f"Cycle Variability (CV): {timing['cycle_time_cv_percent']:.1f}%")

        phase = result['phase_distribution']
        print(f"\nStance Phase: {phase['stance_percent_mean']:.1f} ± {phase['stance_percent_std']:.1f}%")
        print(f"Swing Phase: {phase['swing_percent_mean']:.1f} ± {phase['swing_percent_std']:.1f}%")
        print(f"Double Support: {phase['double_support_percent']:.1f}%")

    print(f"\nEvents Detected: {len(result['events'])}")
    for event in result['events'][:5]:
        print(f"  {event['type']} ({event['side']}) @ {event['time']:.2f}s [conf: {event['confidence']:.2f}]")
    if len(result['events']) > 5:
        print(f"  ... and {len(result['events']) - 5} more")
