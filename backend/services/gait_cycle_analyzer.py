"""
Gait Cycle Analyzer
Detailed gait cycle phase detection and analysis

Gait Cycle Phases (8 sub-phases):
STANCE PHASE (~60% of cycle):
  1. Initial Contact (IC) / Heel Strike - 0%
  2. Loading Response (LR) - 0-10%
  3. Mid Stance (MSt) - 10-30%
  4. Terminal Stance (TSt) - 30-50%
  5. Pre-Swing (PSw) - 50-60%

SWING PHASE (~40% of cycle):
  6. Initial Swing (ISw) - 60-73%
  7. Mid Swing (MSw) - 73-87%
  8. Terminal Swing (TSw) - 87-100%

Reference: Perry & Burnfield (2010) "Gait Analysis: Normal and Pathological Function"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.signal import find_peaks, savgol_filter
import json


@dataclass
class GaitEvent:
    """Single gait event (heel strike or toe off)"""
    frame_idx: int
    timestamp: float
    event_type: str  # 'HS' (heel strike) or 'TO' (toe off)
    side: str  # 'L' (left) or 'R' (right)


@dataclass
class GaitCycle:
    """Complete gait cycle from heel strike to next heel strike (same foot)"""
    cycle_number: int
    side: str  # 'L' or 'R'

    # Events
    initial_contact_frame: int
    initial_contact_time: float
    toe_off_frame: int
    toe_off_time: float
    terminal_swing_frame: int  # Next heel strike
    terminal_swing_time: float

    # Phase durations (seconds)
    stance_time: float
    swing_time: float
    cycle_time: float  # Total gait cycle duration

    # Phase percentages (% of cycle)
    stance_percent: float
    swing_percent: float

    # Sub-phase timing (% of cycle)
    loading_response_percent: float  # 0-10%
    mid_stance_percent: float        # 10-30%
    terminal_stance_percent: float   # 30-50%
    pre_swing_percent: float         # 50-60%
    initial_swing_percent: float     # 60-73%
    mid_swing_percent: float         # 73-87%
    terminal_swing_percent: float    # 87-100%

    # Spatial metrics for this cycle
    step_length: float  # meters
    step_height: float  # meters (foot clearance)
    arm_swing_amplitude: float  # meters

    # Kinematics at key events
    hip_angle_at_ic: float  # Hip flexion at initial contact (degrees)
    knee_angle_at_ic: float
    ankle_angle_at_ic: float
    knee_angle_at_to: float  # Knee flexion at toe off
    peak_knee_flexion_swing: float  # Peak knee flexion during swing


@dataclass
class GaitCycleAnalysis:
    """Complete gait cycle analysis results"""
    # Summary statistics
    num_cycles_left: int
    num_cycles_right: int
    total_cycles: int
    analysis_duration: float  # seconds

    # Cycle timing (mean ± std)
    cycle_time_mean: float
    cycle_time_std: float
    cycle_time_cv: float  # Coefficient of variation (%)

    # Phase timing (mean % of cycle)
    stance_percent_mean: float
    stance_percent_std: float
    swing_percent_mean: float
    swing_percent_std: float

    # Double support (both feet on ground)
    double_support_percent: float
    single_support_percent: float

    # Sub-phase timing variability (CV %)
    loading_response_cv: float
    mid_stance_cv: float
    terminal_stance_cv: float
    pre_swing_cv: float
    initial_swing_cv: float
    mid_swing_cv: float
    terminal_swing_cv: float

    # Asymmetry metrics (L vs R)
    stance_time_asymmetry: float  # %
    swing_time_asymmetry: float
    cycle_time_asymmetry: float
    step_length_asymmetry: float

    # Cycle-to-cycle variability
    step_length_cv: float
    step_time_cv: float
    stride_time_cv: float

    # Individual cycles (for detailed analysis)
    left_cycles: List[GaitCycle] = field(default_factory=list)
    right_cycles: List[GaitCycle] = field(default_factory=list)

    # Gait events timeline
    events: List[GaitEvent] = field(default_factory=list)


class GaitCycleAnalyzer:
    """Analyze gait cycles from pose landmark data"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze(self, landmark_frames: List[Dict]) -> GaitCycleAnalysis:
        """
        Perform detailed gait cycle analysis

        Args:
            landmark_frames: List of frames with pose landmarks

        Returns:
            GaitCycleAnalysis with detailed cycle breakdown
        """
        if len(landmark_frames) < 30:
            raise ValueError("Insufficient data for gait cycle analysis (need at least 30 frames)")

        # Extract trajectories
        trajectories = self._extract_trajectories(landmark_frames)

        # Detect gait events
        events = self._detect_gait_events(trajectories)

        if len(events) < 4:
            raise ValueError("Insufficient gait events detected")

        # Build gait cycles
        left_cycles = self._build_cycles(events, trajectories, side='L')
        right_cycles = self._build_cycles(events, trajectories, side='R')

        # Calculate summary statistics
        analysis = self._calculate_summary(left_cycles, right_cycles, events, trajectories)

        return analysis

    def _extract_trajectories(self, landmark_frames: List[Dict]) -> Dict:
        """Extract landmark trajectories from frames"""
        trajectories = {
            'timestamps': [],
            'left_ankle': [], 'right_ankle': [],
            'left_hip': [], 'right_hip': [],
            'left_knee': [], 'right_knee': [],
            'left_shoulder': [], 'right_shoulder': [],
            'left_wrist': [], 'right_wrist': [],
            'left_toe': [], 'right_toe': [],
            # World landmarks (meters)
            'left_ankle_world': [], 'right_ankle_world': [],
            'left_hip_world': [], 'right_hip_world': [],
            'left_knee_world': [], 'right_knee_world': [],
            'left_shoulder_world': [], 'right_shoulder_world': [],
        }

        landmark_ids = {
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_wrist': 15, 'right_wrist': 16,
            'left_toe': 31, 'right_toe': 32,
        }

        for frame in landmark_frames:
            keypoints = frame.get('keypoints', frame.get('landmarks', []))
            kp_dict = {kp['id']: kp for kp in keypoints}

            world_kp = frame.get('world_keypoints', frame.get('world_landmarks', []))
            world_dict = {kp['id']: kp for kp in world_kp} if world_kp else {}

            # Check required landmarks
            required = [23, 24, 27, 28]
            if not all(k in kp_dict for k in required):
                continue

            ts = frame.get('timestamp', frame.get('frame_number', 0) / self.fps)
            trajectories['timestamps'].append(ts)

            # Extract normalized landmarks
            for name, lid in landmark_ids.items():
                if lid in kp_dict:
                    trajectories[name].append([
                        kp_dict[lid]['x'],
                        kp_dict[lid]['y'],
                        kp_dict[lid].get('z', 0)
                    ])
                else:
                    # Use last value or zero
                    last_val = trajectories[name][-1] if trajectories[name] else [0, 0, 0]
                    trajectories[name].append(last_val)

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
            trajectories[key] = np.array(trajectories[key])

        return trajectories

    def _detect_gait_events(self, trajectories: Dict) -> List[GaitEvent]:
        """Detect heel strikes and toe offs using Zeni method"""
        events = []
        timestamps = trajectories['timestamps']

        if len(timestamps) < 10:
            return events

        # Calculate hip center
        hip_center = (trajectories['left_hip'] + trajectories['right_hip']) / 2

        # Relative foot position (Z-axis = forward/backward)
        # Use world landmarks if available, otherwise normalized
        if len(trajectories.get('left_ankle_world', [])) > 10:
            left_ankle = trajectories['left_ankle_world']
            right_ankle = trajectories['right_ankle_world']
            hip_world = (trajectories['left_hip_world'] + trajectories['right_hip_world']) / 2
            left_rel_z = left_ankle[:, 2] - hip_world[:, 2]
            right_rel_z = right_ankle[:, 2] - hip_world[:, 2]
        else:
            left_rel_z = trajectories['left_ankle'][:, 2] - hip_center[:, 2]
            right_rel_z = trajectories['right_ankle'][:, 2] - hip_center[:, 2]

        # Smooth signals
        if len(left_rel_z) > 15:
            left_rel_z = savgol_filter(left_rel_z, 15, 3)
            right_rel_z = savgol_filter(right_rel_z, 15, 3)

        min_stride_frames = int(self.fps * 0.35)  # Min 350ms between same-foot events

        # Detect heel strikes (foot furthest forward = local max)
        left_hs, _ = find_peaks(left_rel_z, distance=min_stride_frames,
                                prominence=0.02 * np.ptp(left_rel_z))
        right_hs, _ = find_peaks(right_rel_z, distance=min_stride_frames,
                                 prominence=0.02 * np.ptp(right_rel_z))

        # Detect toe offs (foot furthest backward = local min)
        left_to, _ = find_peaks(-left_rel_z, distance=min_stride_frames,
                                prominence=0.02 * np.ptp(left_rel_z))
        right_to, _ = find_peaks(-right_rel_z, distance=min_stride_frames,
                                 prominence=0.02 * np.ptp(right_rel_z))

        # Create event list
        for idx in left_hs:
            events.append(GaitEvent(
                frame_idx=int(idx),
                timestamp=float(timestamps[idx]),
                event_type='HS',
                side='L'
            ))

        for idx in right_hs:
            events.append(GaitEvent(
                frame_idx=int(idx),
                timestamp=float(timestamps[idx]),
                event_type='HS',
                side='R'
            ))

        for idx in left_to:
            events.append(GaitEvent(
                frame_idx=int(idx),
                timestamp=float(timestamps[idx]),
                event_type='TO',
                side='L'
            ))

        for idx in right_to:
            events.append(GaitEvent(
                frame_idx=int(idx),
                timestamp=float(timestamps[idx]),
                event_type='TO',
                side='R'
            ))

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    def _build_cycles(self, events: List[GaitEvent], trajectories: Dict,
                      side: str) -> List[GaitCycle]:
        """Build complete gait cycles for one side"""
        cycles = []

        # Filter events for this side
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
                continue

            to_event = to_between[0]

            # Calculate timing
            stance_time = to_event.timestamp - ic_event.timestamp
            swing_time = next_ic_event.timestamp - to_event.timestamp
            cycle_time = next_ic_event.timestamp - ic_event.timestamp

            if cycle_time <= 0 or stance_time <= 0 or swing_time <= 0:
                continue

            stance_percent = (stance_time / cycle_time) * 100
            swing_percent = (swing_time / cycle_time) * 100

            # Sub-phase percentages (normalized to typical gait)
            # These are estimates based on Perry & Burnfield
            loading_response_percent = min(10.0, stance_percent * 0.17)
            mid_stance_percent = stance_percent * 0.33
            terminal_stance_percent = stance_percent * 0.33
            pre_swing_percent = stance_percent * 0.17
            initial_swing_percent = swing_percent * 0.33
            mid_swing_percent = swing_percent * 0.34
            terminal_swing_percent = swing_percent * 0.33

            # Calculate step length (using hip-width scaling)
            step_length = self._calculate_step_length(
                trajectories, ic_event.frame_idx, next_ic_event.frame_idx, side)

            # Calculate step height
            step_height = self._calculate_step_height(
                trajectories, to_event.frame_idx, next_ic_event.frame_idx, side)

            # Calculate arm swing amplitude for this cycle
            arm_swing = self._calculate_arm_swing(
                trajectories, ic_event.frame_idx, next_ic_event.frame_idx, side)

            # Joint angles (if available)
            hip_ic, knee_ic, ankle_ic = self._get_joint_angles_at_frame(
                trajectories, ic_event.frame_idx, side)
            _, knee_to, _ = self._get_joint_angles_at_frame(
                trajectories, to_event.frame_idx, side)
            peak_knee_swing = self._get_peak_knee_flexion(
                trajectories, to_event.frame_idx, next_ic_event.frame_idx, side)

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
                peak_knee_flexion_swing=float(peak_knee_swing)
            )

            cycles.append(cycle)

        return cycles

    def _calculate_step_length(self, trajectories: Dict, start_idx: int,
                               end_idx: int, side: str) -> float:
        """Calculate step length for a gait cycle"""
        ankle_key = f"{side.lower()}eft_ankle" if side == 'L' else "right_ankle"
        if side == 'L':
            ankle_key = 'left_ankle'
        else:
            ankle_key = 'right_ankle'

        if start_idx >= len(trajectories[ankle_key]) or end_idx >= len(trajectories[ankle_key]):
            return 0.0

        start_pos = trajectories[ankle_key][start_idx]
        end_pos = trajectories[ankle_key][end_idx]

        # Distance in normalized coordinates
        dist_norm = np.linalg.norm(end_pos - start_pos)

        # Scale to meters using hip width
        left_hip = trajectories['left_hip']
        right_hip = trajectories['right_hip']
        hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
        avg_hip_width = np.mean(hip_widths[hip_widths > 0.01])

        REAL_HIP_WIDTH = 0.30  # meters
        scale = REAL_HIP_WIDTH / avg_hip_width if avg_hip_width > 0 else 2.0
        scale = np.clip(scale, 1.0, 15.0)

        step_length = dist_norm * scale
        return step_length if 0.1 < step_length < 2.0 else 0.0

    def _calculate_step_height(self, trajectories: Dict, to_idx: int,
                               hs_idx: int, side: str) -> float:
        """Calculate foot clearance during swing phase"""
        world_key = 'left_ankle_world' if side == 'L' else 'right_ankle_world'

        if world_key not in trajectories or len(trajectories[world_key]) == 0:
            return 0.0

        ankle_world = trajectories[world_key]

        if to_idx >= len(ankle_world) or hs_idx >= len(ankle_world):
            return 0.0

        # Get Y values during swing (to_idx to hs_idx)
        if hs_idx <= to_idx:
            return 0.0

        swing_y = ankle_world[to_idx:hs_idx, 1]
        if len(swing_y) < 3:
            return 0.0

        baseline_y = min(ankle_world[to_idx, 1], ankle_world[min(hs_idx, len(ankle_world)-1), 1])
        peak_y = np.max(swing_y)
        step_height = peak_y - baseline_y

        return step_height if 0.01 < step_height < 0.50 else 0.0

    def _calculate_arm_swing(self, trajectories: Dict, start_idx: int,
                             end_idx: int, side: str) -> float:
        """Calculate arm swing amplitude for a cycle"""
        # Use opposite arm (contralateral arm swing)
        wrist_key = 'right_wrist' if side == 'L' else 'left_wrist'

        if start_idx >= len(trajectories[wrist_key]) or end_idx >= len(trajectories[wrist_key]):
            return 0.0

        wrist_segment = trajectories[wrist_key][start_idx:end_idx]
        if len(wrist_segment) < 5:
            return 0.0

        # Calculate amplitude as peak-to-peak in X direction (frontal)
        amplitude_norm = np.ptp(wrist_segment[:, 0])

        # Scale to meters
        left_hip = trajectories['left_hip']
        right_hip = trajectories['right_hip']
        hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
        avg_hip_width = np.mean(hip_widths[hip_widths > 0.01])

        REAL_HIP_WIDTH = 0.30
        scale = REAL_HIP_WIDTH / avg_hip_width if avg_hip_width > 0 else 2.0

        return amplitude_norm * scale

    def _get_joint_angles_at_frame(self, trajectories: Dict, frame_idx: int,
                                   side: str) -> Tuple[float, float, float]:
        """Get hip, knee, ankle angles at a specific frame"""
        prefix = 'left' if side == 'L' else 'right'

        # Check if we have world landmarks
        hip_key = f"{prefix}_hip_world"
        knee_key = f"{prefix}_knee_world"
        ankle_key = f"{prefix}_ankle"
        shoulder_key = f"{prefix}_shoulder"

        # Use normalized landmarks as fallback
        if hip_key not in trajectories or len(trajectories[hip_key]) <= frame_idx:
            return 0.0, 0.0, 0.0

        hip = trajectories[hip_key][frame_idx] if len(trajectories.get(hip_key, [])) > frame_idx else None
        knee = trajectories.get(f"{prefix}_knee_world", trajectories.get(f"{prefix}_knee", []))
        knee = knee[frame_idx] if len(knee) > frame_idx else None
        ankle = trajectories.get(f"{prefix}_ankle_world", trajectories.get(f"{prefix}_ankle", []))
        ankle = ankle[frame_idx] if len(ankle) > frame_idx else None

        if hip is None or knee is None or ankle is None:
            return 0.0, 0.0, 0.0

        hip = np.array(hip)
        knee = np.array(knee)
        ankle = np.array(ankle)

        # Hip flexion: angle at hip between vertical and thigh
        thigh_vec = knee - hip
        vertical = np.array([0, -1, 0])  # Y down in normalized coords
        hip_angle = self._angle_between(thigh_vec, vertical)

        # Knee flexion: angle at knee between thigh and shank
        shank_vec = ankle - knee
        knee_angle = 180 - self._angle_between(-thigh_vec, shank_vec)

        # Ankle angle (simplified)
        ankle_angle = 90.0  # Placeholder

        return hip_angle, knee_angle, ankle_angle

    def _get_peak_knee_flexion(self, trajectories: Dict, to_idx: int,
                               hs_idx: int, side: str) -> float:
        """Get peak knee flexion during swing phase"""
        prefix = 'left' if side == 'L' else 'right'

        hip_key = f"{prefix}_hip_world"
        knee_key = f"{prefix}_knee_world"
        ankle_key = f"{prefix}_ankle_world"

        if hip_key not in trajectories:
            return 0.0

        hip_arr = trajectories.get(hip_key, [])
        knee_arr = trajectories.get(knee_key, [])
        ankle_arr = trajectories.get(ankle_key, [])

        if len(hip_arr) <= hs_idx or len(knee_arr) <= hs_idx or len(ankle_arr) <= hs_idx:
            return 0.0

        peak_flexion = 0.0
        for i in range(to_idx, min(hs_idx, len(hip_arr))):
            hip = np.array(hip_arr[i])
            knee = np.array(knee_arr[i])
            ankle = np.array(ankle_arr[i])

            thigh_vec = knee - hip
            shank_vec = ankle - knee
            knee_angle = 180 - self._angle_between(-thigh_vec, shank_vec)

            if knee_angle > peak_flexion:
                peak_flexion = knee_angle

        return peak_flexion

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _calculate_summary(self, left_cycles: List[GaitCycle],
                          right_cycles: List[GaitCycle],
                          events: List[GaitEvent],
                          trajectories: Dict) -> GaitCycleAnalysis:
        """Calculate summary statistics from all cycles"""
        all_cycles = left_cycles + right_cycles

        if not all_cycles:
            raise ValueError("No complete gait cycles detected")

        timestamps = trajectories['timestamps']
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0

        # Cycle timing statistics
        cycle_times = [c.cycle_time for c in all_cycles]
        cycle_time_mean = float(np.mean(cycle_times))
        cycle_time_std = float(np.std(cycle_times))
        cycle_time_cv = (cycle_time_std / cycle_time_mean * 100) if cycle_time_mean > 0 else 0

        # Phase timing
        stance_percents = [c.stance_percent for c in all_cycles]
        swing_percents = [c.swing_percent for c in all_cycles]

        stance_percent_mean = float(np.mean(stance_percents))
        stance_percent_std = float(np.std(stance_percents))
        swing_percent_mean = float(np.mean(swing_percents))
        swing_percent_std = float(np.std(swing_percents))

        # Double support estimation
        # DS = 100% - (left_single_support + right_single_support)
        # Single support ≈ swing time of opposite foot
        double_support_percent = max(0, 100 - swing_percent_mean * 2)
        single_support_percent = 100 - double_support_percent

        # Sub-phase variability
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

        # Asymmetry calculations
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

        # Cycle-to-cycle variability
        step_lengths = [c.step_length for c in all_cycles if c.step_length > 0]
        step_length_cv = calc_cv(step_lengths)

        step_times = [c.stance_time for c in all_cycles]
        step_time_cv = calc_cv(step_times)

        stride_time_cv = cycle_time_cv  # Same as cycle time CV

        return GaitCycleAnalysis(
            num_cycles_left=len(left_cycles),
            num_cycles_right=len(right_cycles),
            total_cycles=len(all_cycles),
            analysis_duration=float(duration),
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
            events=events
        )

    def to_dict(self, analysis: GaitCycleAnalysis) -> Dict:
        """Convert analysis to dictionary for JSON serialization"""
        return {
            'summary': {
                'num_cycles_left': analysis.num_cycles_left,
                'num_cycles_right': analysis.num_cycles_right,
                'total_cycles': analysis.total_cycles,
                'analysis_duration_sec': analysis.analysis_duration,
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
                    'side': e.side
                }
                for e in analysis.events
            ]
        }

    def _cycle_to_dict(self, cycle: GaitCycle) -> Dict:
        """Convert single cycle to dictionary"""
        return {
            'cycle_number': cycle.cycle_number,
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


def analyze_gait_cycles_from_file(skeleton_json_path: str, fps: float = 30.0) -> Dict:
    """
    Analyze gait cycles from a skeleton JSON file

    Args:
        skeleton_json_path: Path to skeleton landmarks JSON
        fps: Video frame rate

    Returns:
        Dictionary with gait cycle analysis results
    """
    with open(skeleton_json_path, 'r') as f:
        landmark_frames = json.load(f)

    analyzer = GaitCycleAnalyzer(fps=fps)
    analysis = analyzer.analyze(landmark_frames)
    return analyzer.to_dict(analysis)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gait_cycle_analyzer.py <skeleton_json> [fps]")
        sys.exit(1)

    skeleton_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    print(f"Analyzing gait cycles from: {skeleton_path}")
    print(f"FPS: {fps}")

    result = analyze_gait_cycles_from_file(skeleton_path, fps)

    print("\n" + "="*60)
    print("GAIT CYCLE ANALYSIS RESULTS")
    print("="*60)

    summary = result['summary']
    print(f"\nTotal Cycles: {summary['total_cycles']} (L:{summary['num_cycles_left']}, R:{summary['num_cycles_right']})")
    print(f"Duration: {summary['analysis_duration_sec']:.1f} sec")

    timing = result['timing']
    print(f"\nCycle Time: {timing['cycle_time_mean_sec']:.3f} ± {timing['cycle_time_std_sec']:.3f} sec")
    print(f"Cycle Variability (CV): {timing['cycle_time_cv_percent']:.1f}%")

    phase = result['phase_distribution']
    print(f"\nStance Phase: {phase['stance_percent_mean']:.1f} ± {phase['stance_percent_std']:.1f}%")
    print(f"Swing Phase: {phase['swing_percent_mean']:.1f} ± {phase['swing_percent_std']:.1f}%")
    print(f"Double Support: {phase['double_support_percent']:.1f}%")

    asym = result['asymmetry_percent']
    print(f"\nAsymmetry:")
    print(f"  Stance Time: {asym['stance_time']:.1f}%")
    print(f"  Swing Time: {asym['swing_time']:.1f}%")
    print(f"  Step Length: {asym['step_length']:.1f}%")

    var = result['variability_cv']
    print(f"\nVariability (CV):")
    print(f"  Step Length: {var['step_length']:.1f}%")
    print(f"  Step Time: {var['step_time']:.1f}%")
    print(f"  Stride Time: {var['stride_time']:.1f}%")

    # Print first cycle details
    if result['cycles']['left']:
        print(f"\n--- Sample Left Cycle (#1) ---")
        c = result['cycles']['left'][0]
        print(f"Duration: {c['durations_sec']['total']:.3f} sec")
        print(f"Stance/Swing: {c['phase_percent']['stance']:.1f}% / {c['phase_percent']['swing']:.1f}%")
        print(f"Step Length: {c['spatial_meters']['step_length']:.3f} m")
        print(f"Step Height: {c['spatial_meters']['step_height']:.3f} m")
