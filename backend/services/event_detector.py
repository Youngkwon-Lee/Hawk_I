"""
Event Detector Service
Detects timeline events from analysis data
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TimelineEvent:
    """Single timeline event"""
    timestamp: float  # seconds
    type: str  # event type
    description: str  # Korean description
    severity: str  # info, warning, critical


class EventDetector:
    """
    Detect timeline events from metrics and skeleton data
    
    Supports:
    - Finger Tapping: start, speed changes, fatigue, hesitation, end
    - Gait: freezing, turns
    """
    
    def detect_events(
        self,
        video_type: str,
        metrics: Dict[str, Any],
        landmark_frames: List[Any],
        fps: float
    ) -> List[Dict[str, Any]]:
        """
        Detect events based on video type
        
        Args:
            video_type: Type of analysis (finger_tapping, gait, etc.)
            metrics: Calculated metrics
            landmark_frames: Skeleton data
            fps: Video frame rate
            
        Returns:
            List of event dictionaries
        """
        if video_type == "finger_tapping":
            events = self._detect_tapping_events(metrics, landmark_frames, fps)
        elif video_type == "gait":
            events = self._detect_gait_events(metrics, landmark_frames, fps)
        else:
            events = []
        
        # Convert to dict format
        return [
            {
                "timestamp": e.timestamp,
                "type": e.type,
                "description": e.description,
                "severity": e.severity
            }
            for e in events
        ]
    
    def _detect_tapping_events(
        self,
        metrics: Dict[str, Any],
        landmark_frames: List[Any],
        fps: float
    ) -> List[TimelineEvent]:
        """Detect finger tapping events"""
        events = []
        
        if not metrics or not landmark_frames:
            return events
        
        duration = metrics.get("duration", 0)
        
        # Start event
        events.append(TimelineEvent(
            timestamp=0.0,
            type="start",
            description="시작",
            severity="info"
        ))
        
        # Speed change detection (based on rhythm variability)
        rhythm_var = metrics.get("rhythm_variability", 0)
        if rhythm_var > 25:  # High variability indicates speed changes
            # Add event at 1/3 duration
            events.append(TimelineEvent(
                timestamp=duration * 0.33,
                type="speed_change",
                description="속도 변화",
                severity="warning"
            ))
        
        # Fatigue detection
        fatigue_rate = metrics.get("fatigue_rate", 0)
        if fatigue_rate > 30:  # Significant fatigue
            # Add event at 2/3 duration (when fatigue typically appears)
            events.append(TimelineEvent(
                timestamp=duration * 0.67,
                type="fatigue",
                description="피로 감지",
                severity="warning"
            ))
        
        # Hesitation events
        hesitation_count = metrics.get("hesitation_count", 0)
        if hesitation_count > 0:
            # Add hesitation event at mid-point
            events.append(TimelineEvent(
                timestamp=duration * 0.5,
                type="hesitation",
                description=f"주저함 ({hesitation_count}회)",
                severity="info"
            ))
        
        # End event
        events.append(TimelineEvent(
            timestamp=duration,
            type="end",
            description="종료",
            severity="info"
        ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def _detect_gait_events(
        self,
        metrics: Dict[str, Any],
        landmark_frames: List[Any],
        fps: float
    ) -> List[TimelineEvent]:
        """Detect gait events (freezing and turns only)"""
        events = []
        
        if not landmark_frames:
            return events
        
        duration = len(landmark_frames) / fps
        
        # Start event
        events.append(TimelineEvent(
            timestamp=0.0,
            type="start",
            description="걷기 시작",
            severity="info"
        ))
        
        # Detect turns from pelvis/hip rotation
        turn_events = self._detect_turns(landmark_frames, fps)
        events.extend(turn_events)
        
        # Detect freezing from velocity drops
        freezing_events = self._detect_freezing(landmark_frames, fps)
        events.extend(freezing_events)
        
        # End event
        events.append(TimelineEvent(
            timestamp=duration,
            type="end",
            description="걷기 종료",
            severity="info"
        ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def _detect_turns(
        self,
        landmark_frames: List[Any],
        fps: float
    ) -> List[TimelineEvent]:
        """Detect turn events from pelvis rotation"""
        events = []
        
        # Calculate pelvis yaw angle changes
        yaw_angles = []
        for frame in landmark_frames:
            # Get hip landmarks (left hip: 23, right hip: 24)
            landmarks = frame.landmarks
            if len(landmarks) > 24:
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                # Calculate yaw from hip positions
                dx = right_hip['x'] - left_hip['x']
                dy = right_hip['y'] - left_hip['y']
                yaw = np.arctan2(dy, dx) * 180 / np.pi
                yaw_angles.append(yaw)
        
        if len(yaw_angles) < 10:
            return events
        
        # Detect significant yaw changes (turns)
        window_size = int(fps * 0.5)  # 0.5 second window
        for i in range(window_size, len(yaw_angles) - window_size):
            prev_yaw = np.mean(yaw_angles[i-window_size:i])
            curr_yaw = np.mean(yaw_angles[i:i+window_size])
            
            yaw_change = abs(curr_yaw - prev_yaw)
            
            # Turn detected if yaw change > 30 degrees
            if yaw_change > 30:
                timestamp = i / fps
                events.append(TimelineEvent(
                    timestamp=timestamp,
                    type="turn",
                    description="턴 감지",
                    severity="info"
                ))
                # Skip ahead to avoid duplicate detections
                i += window_size
        
        return events
    
    def _detect_freezing(
        self,
        landmark_frames: List[Any],
        fps: float
    ) -> List[TimelineEvent]:
        """Detect freezing events from velocity drops"""
        events = []
        
        # Calculate center of mass velocity
        velocities = []
        prev_com = None
        
        for frame in landmark_frames:
            landmarks = frame.landmarks
            if len(landmarks) > 24:
                # Calculate center of mass from hips
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                com_x = (left_hip['x'] + right_hip['x']) / 2
                com_y = (left_hip['y'] + right_hip['y']) / 2
                
                if prev_com is not None:
                    dx = com_x - prev_com[0]
                    dy = com_y - prev_com[1]
                    velocity = np.sqrt(dx**2 + dy**2) * fps
                    velocities.append(velocity)
                
                prev_com = (com_x, com_y)
        
        if len(velocities) < 10:
            return events
        
        # Detect freezing (velocity drops below 20% of mean)
        mean_velocity = np.mean(velocities)
        threshold = mean_velocity * 0.2
        
        window_size = int(fps * 0.3)  # 0.3 second window
        in_freeze = False
        freeze_start = 0
        
        for i in range(window_size, len(velocities)):
            window_vel = np.mean(velocities[i-window_size:i])
            
            if window_vel < threshold and not in_freeze:
                # Freezing started
                in_freeze = True
                freeze_start = i
            elif window_vel >= threshold and in_freeze:
                # Freezing ended
                in_freeze = False
                timestamp = freeze_start / fps
                events.append(TimelineEvent(
                    timestamp=timestamp,
                    type="freezing",
                    description="동결 감지",
                    severity="warning"
                ))
        
        return events
