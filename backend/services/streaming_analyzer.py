"""
Streaming Analyzer - Real-time Video Analysis

Provides real-time analysis of video frames with progressive score updates.
Designed for WebSocket integration and live feedback.

Features:
- Frame-by-frame landmark extraction
- Rolling window metric calculation
- Progressive UPDRS score updates
- Event detection (hesitations, halts, freezing)
- Low-latency feedback loop

Usage:
    analyzer = StreamingAnalyzer(task_type='finger_tapping')
    analyzer.start_session()

    for frame in video_stream:
        result = analyzer.process_frame(frame)
        send_to_client(result)

    final_result = analyzer.end_session()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import time
import cv2

# Import MediaPipe components
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


@dataclass
class FrameResult:
    """Result from processing a single frame"""
    frame_number: int
    timestamp: float
    landmarks_detected: bool
    landmark_count: int
    confidence: float

    # Real-time metrics (rolling window)
    current_speed: Optional[float] = None
    current_amplitude: Optional[float] = None
    current_rhythm: Optional[float] = None

    # Progressive score (updated every N frames)
    progressive_score: Optional[float] = None
    score_confidence: Optional[float] = None

    # Events detected in this frame
    events: List[str] = field(default_factory=list)

    # Raw landmarks (for visualization)
    landmarks: Optional[List[Dict]] = None


@dataclass
class SessionResult:
    """Final result after session ends"""
    session_id: str
    total_frames: int
    processed_frames: int
    duration_seconds: float
    task_type: str

    # Final metrics
    final_score: float
    final_confidence: float
    severity: str

    # Aggregated metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Event summary
    event_counts: Dict[str, int] = field(default_factory=dict)

    # Score progression (for visualization)
    score_history: List[Tuple[float, float]] = field(default_factory=list)


class StreamingAnalyzer:
    """Real-time streaming video analyzer"""

    # Configuration
    WINDOW_SIZE = 30           # Frames for rolling calculations
    SCORE_UPDATE_INTERVAL = 15  # Update score every N frames
    MIN_FRAMES_FOR_SCORE = 20   # Minimum frames before first score

    def __init__(self, task_type: str = 'finger_tapping', fps: float = 30.0):
        self.task_type = task_type
        self.fps = fps
        self.dt = 1.0 / fps

        # Session state
        self.session_id: Optional[str] = None
        self.session_start: Optional[float] = None
        self.frame_count = 0
        self.processed_count = 0

        # Rolling buffers
        self.landmark_buffer: deque = deque(maxlen=self.WINDOW_SIZE)
        self.position_buffer: deque = deque(maxlen=self.WINDOW_SIZE)
        self.velocity_buffer: deque = deque(maxlen=self.WINDOW_SIZE)
        self.amplitude_buffer: deque = deque(maxlen=self.WINDOW_SIZE * 2)

        # Score progression
        self.score_history: List[Tuple[float, float]] = []
        self.last_score: float = 0.0
        self.last_confidence: float = 0.0

        # Event counters
        self.event_counts: Dict[str, int] = {
            'hesitation': 0,
            'halt': 0,
            'freeze': 0,
            'tremor': 0
        }

        # MediaPipe processors
        self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe hands/pose detector"""
        if not MP_AVAILABLE:
            self.hands = None
            self.pose = None
            return

        if self.task_type in ['finger_tapping', 'hand_movement']:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = None
        else:
            self.hands = None
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new analysis session"""
        import uuid
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_start = time.time()
        self.frame_count = 0
        self.processed_count = 0

        # Clear buffers
        self.landmark_buffer.clear()
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.amplitude_buffer.clear()

        # Reset counters
        self.score_history.clear()
        self.event_counts = {k: 0 for k in self.event_counts}
        self.last_score = 0.0
        self.last_confidence = 0.0

        return self.session_id

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single video frame

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            FrameResult with real-time metrics
        """
        self.frame_count += 1
        timestamp = (self.frame_count - 1) * self.dt

        result = FrameResult(
            frame_number=self.frame_count,
            timestamp=timestamp,
            landmarks_detected=False,
            landmark_count=0,
            confidence=0.0
        )

        # Extract landmarks
        landmarks = self._extract_landmarks(frame)

        if landmarks:
            result.landmarks_detected = True
            result.landmark_count = len(landmarks)
            result.confidence = self._calculate_confidence(landmarks)
            result.landmarks = landmarks

            self.processed_count += 1
            self.landmark_buffer.append(landmarks)

            # Calculate real-time metrics
            position = self._get_key_position(landmarks)
            if position is not None:
                self.position_buffer.append(position)

                # Calculate velocity
                if len(self.position_buffer) >= 2:
                    velocity = np.array(self.position_buffer[-1]) - np.array(self.position_buffer[-2])
                    velocity = velocity / self.dt
                    self.velocity_buffer.append(velocity)
                    result.current_speed = float(np.linalg.norm(velocity))

                # Calculate amplitude (for tapping)
                if self.task_type in ['finger_tapping', 'hand_movement']:
                    amplitude = self._calculate_amplitude(landmarks)
                    if amplitude is not None:
                        self.amplitude_buffer.append(amplitude)
                        result.current_amplitude = amplitude

            # Detect events
            events = self._detect_events()
            result.events = events
            for event in events:
                if event in self.event_counts:
                    self.event_counts[event] += 1

            # Update progressive score
            if self.processed_count >= self.MIN_FRAMES_FOR_SCORE:
                if self.processed_count % self.SCORE_UPDATE_INTERVAL == 0:
                    score, confidence = self._calculate_progressive_score()
                    result.progressive_score = score
                    result.score_confidence = confidence
                    self.last_score = score
                    self.last_confidence = confidence
                    self.score_history.append((timestamp, score))
                else:
                    result.progressive_score = self.last_score
                    result.score_confidence = self.last_confidence

            # Calculate rhythm variability
            if len(self.velocity_buffer) >= 5:
                speeds = [np.linalg.norm(v) for v in self.velocity_buffer]
                result.current_rhythm = float(np.std(speeds) / (np.mean(speeds) + 1e-6))

        return result

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """Extract landmarks from frame using MediaPipe"""
        if not MP_AVAILABLE:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.hands:
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = []
                for i, lm in enumerate(hand.landmark):
                    landmarks.append({
                        'id': i,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': 1.0
                    })
                return landmarks

        elif self.pose:
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                landmarks = []
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        'id': i,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })
                return landmarks

        return None

    def _calculate_confidence(self, landmarks: List[Dict]) -> float:
        """Calculate average landmark confidence"""
        if not landmarks:
            return 0.0
        visibilities = [lm.get('visibility', 1.0) for lm in landmarks]
        return float(np.mean(visibilities))

    def _get_key_position(self, landmarks: List[Dict]) -> Optional[List[float]]:
        """Get key landmark position for tracking"""
        if not landmarks:
            return None

        if self.task_type in ['finger_tapping', 'hand_movement']:
            # Use index finger tip (landmark 8)
            if len(landmarks) > 8:
                lm = landmarks[8]
                return [lm['x'], lm['y'], lm.get('z', 0)]
        else:
            # Use hip center for gait
            if len(landmarks) > 24:
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                return [
                    (left_hip['x'] + right_hip['x']) / 2,
                    (left_hip['y'] + right_hip['y']) / 2,
                    (left_hip.get('z', 0) + right_hip.get('z', 0)) / 2
                ]

        return None

    def _calculate_amplitude(self, landmarks: List[Dict]) -> Optional[float]:
        """Calculate finger tapping amplitude"""
        if len(landmarks) < 12:
            return None

        # Distance between thumb tip (4) and index tip (8)
        thumb = landmarks[4]
        index = landmarks[8]

        distance = np.sqrt(
            (thumb['x'] - index['x']) ** 2 +
            (thumb['y'] - index['y']) ** 2
        )

        return float(distance)

    def _detect_events(self) -> List[str]:
        """Detect movement events in current frame"""
        events = []

        if len(self.velocity_buffer) < 3:
            return events

        # Get recent velocities
        recent_speeds = [np.linalg.norm(v) for v in list(self.velocity_buffer)[-5:]]

        # Hesitation: sudden speed decrease
        if len(recent_speeds) >= 3:
            if recent_speeds[-1] < 0.3 * np.mean(recent_speeds[:-1]):
                events.append('hesitation')

        # Halt: very low speed for multiple frames
        if len(recent_speeds) >= 3 and all(s < 0.01 for s in recent_speeds[-3:]):
            events.append('halt')

        # Freeze: no movement detected
        if len(self.position_buffer) >= 5:
            positions = list(self.position_buffer)[-5:]
            total_movement = sum(
                np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                for i in range(1, len(positions))
            )
            if total_movement < 0.001:
                events.append('freeze')

        return events

    def _calculate_progressive_score(self) -> Tuple[float, float]:
        """Calculate progressive UPDRS score from accumulated data"""
        if self.processed_count < self.MIN_FRAMES_FOR_SCORE:
            return 0.0, 0.0

        # Calculate metrics from buffers
        if not self.velocity_buffer:
            return 2.0, 0.5  # Default moderate

        speeds = [np.linalg.norm(v) for v in self.velocity_buffer]
        mean_speed = np.mean(speeds)
        speed_cv = np.std(speeds) / (mean_speed + 1e-6)

        # Amplitude metrics (for finger tapping)
        amplitude_mean = 0.1
        amplitude_cv = 0.3
        if self.amplitude_buffer:
            amplitude_mean = np.mean(list(self.amplitude_buffer))
            amplitude_cv = np.std(list(self.amplitude_buffer)) / (amplitude_mean + 1e-6)

        # Event penalty
        total_events = sum(self.event_counts.values())
        event_rate = total_events / max(self.processed_count, 1)

        # Heuristic scoring (simplified)
        # Lower speed, higher variability, more events = higher UPDRS

        # Speed factor (normalized)
        if self.task_type in ['finger_tapping', 'hand_movement']:
            speed_factor = 1.0 - min(mean_speed * 20, 1.0)  # Faster is better
        else:
            speed_factor = max(0, 1.0 - mean_speed * 2)  # Slower gait = higher score

        # Variability factor
        variability_factor = min(speed_cv, 1.0)

        # Amplitude factor
        amplitude_factor = 1.0 - min(amplitude_mean * 5, 1.0)

        # Event factor
        event_factor = min(event_rate * 10, 1.0)

        # Combine factors
        raw_score = (
            0.3 * speed_factor +
            0.25 * variability_factor +
            0.25 * amplitude_factor +
            0.2 * event_factor
        ) * 4  # Scale to 0-4

        score = np.clip(raw_score, 0, 4)

        # Confidence based on data quality
        confidence = min(self.processed_count / 60, 1.0) * 0.8 + 0.2

        return float(score), float(confidence)

    def end_session(self) -> SessionResult:
        """End the analysis session and return final results"""
        duration = time.time() - self.session_start if self.session_start else 0.0

        # Calculate final score
        final_score, final_confidence = self._calculate_progressive_score()

        # Determine severity
        if final_score < 0.5:
            severity = "Normal"
        elif final_score < 1.5:
            severity = "Slight"
        elif final_score < 2.5:
            severity = "Mild"
        elif final_score < 3.5:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Aggregate metrics
        metrics = {}
        if self.velocity_buffer:
            speeds = [np.linalg.norm(v) for v in self.velocity_buffer]
            metrics['speed_mean'] = float(np.mean(speeds))
            metrics['speed_std'] = float(np.std(speeds))
            metrics['speed_cv'] = float(np.std(speeds) / (np.mean(speeds) + 1e-6))

        if self.amplitude_buffer:
            metrics['amplitude_mean'] = float(np.mean(list(self.amplitude_buffer)))
            metrics['amplitude_std'] = float(np.std(list(self.amplitude_buffer)))

        return SessionResult(
            session_id=self.session_id or "unknown",
            total_frames=self.frame_count,
            processed_frames=self.processed_count,
            duration_seconds=duration,
            task_type=self.task_type,
            final_score=round(final_score, 2),
            final_confidence=round(final_confidence, 3),
            severity=severity,
            metrics=metrics,
            event_counts=self.event_counts.copy(),
            score_history=self.score_history.copy()
        )

    def close(self):
        """Release resources"""
        if self.hands:
            self.hands.close()
        if self.pose:
            self.pose.close()


# WebSocket handler helper
def create_frame_message(result: FrameResult) -> Dict:
    """Create JSON message for WebSocket transmission"""
    return {
        'type': 'frame_result',
        'data': {
            'frame': result.frame_number,
            'timestamp': round(result.timestamp, 3),
            'detected': result.landmarks_detected,
            'confidence': round(result.confidence, 3),
            'metrics': {
                'speed': round(result.current_speed, 4) if result.current_speed else None,
                'amplitude': round(result.current_amplitude, 4) if result.current_amplitude else None,
                'rhythm': round(result.current_rhythm, 4) if result.current_rhythm else None,
            },
            'score': round(result.progressive_score, 2) if result.progressive_score else None,
            'score_confidence': round(result.score_confidence, 3) if result.score_confidence else None,
            'events': result.events
        }
    }


def create_session_message(result: SessionResult) -> Dict:
    """Create JSON message for session end"""
    return {
        'type': 'session_result',
        'data': {
            'session_id': result.session_id,
            'total_frames': result.total_frames,
            'processed_frames': result.processed_frames,
            'duration': round(result.duration_seconds, 2),
            'task_type': result.task_type,
            'final_score': result.final_score,
            'confidence': result.final_confidence,
            'severity': result.severity,
            'metrics': result.metrics,
            'events': result.event_counts,
            'score_history': [
                {'time': round(t, 2), 'score': round(s, 2)}
                for t, s in result.score_history
            ]
        }
    }


if __name__ == "__main__":
    # Test streaming analyzer
    print("Testing StreamingAnalyzer...")

    analyzer = StreamingAnalyzer(task_type='finger_tapping', fps=30)
    session_id = analyzer.start_session()
    print(f"Session started: {session_id}")

    # Simulate 60 frames
    for i in range(60):
        # Create fake frame (black image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.process_frame(frame)

        if i % 15 == 0:
            print(f"Frame {result.frame_number}: detected={result.landmarks_detected}, "
                  f"score={result.progressive_score}")

    final = analyzer.end_session()
    print(f"\nSession ended:")
    print(f"  Frames: {final.processed_frames}/{final.total_frames}")
    print(f"  Duration: {final.duration_seconds:.2f}s")
    print(f"  Final Score: {final.final_score} ({final.severity})")
    print(f"  Confidence: {final.final_confidence}")

    analyzer.close()
