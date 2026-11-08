"""Research-only timing regressions. No videos, clinical scores or personal data."""
import numpy as np
import pytest

from services.metrics_calculator import MetricsCalculator


def _ramp_landmarks(fps=80, n_frames=400, stride=1):
    """A synthetic linearly increasing thumb-index gap over physical time."""
    result = []
    for frame_number in range(0, n_frames, stride):
        t = frame_number / fps
        point = lambda i, x, y: {"id": i, "x": x, "y": y, "z": 0.0, "visibility": 1.0}
        result.append({
            "frame_number": frame_number,
            "timestamp": t,
            "landmarks": [
                point(0, .50, .75),
                point(4, .55 - .01 * t, .40),
                point(5, .65, .60),
                point(6, .65, .53),
                point(7, .65, .46),
                point(8, .65, .40),
                point(9, .55, .60),
                point(13, .48, .62),
                point(17, .42, .63),
            ],
        })
    return result


def test_retained_frame_timestamps_preserve_velocity_at_80fps_and_skip3():
    calc = MetricsCalculator(fps=80.0)
    full = calc.calculate_finger_tapping_metrics(_ramp_landmarks(stride=1))
    sampled = calc.calculate_finger_tapping_metrics(_ramp_landmarks(stride=3))
    # Finger length = 0.20 (in coordinate units), gap increases by 0.01/s.
    expected = 0.05
    assert full.dist_velocity_mean == pytest.approx(expected, rel=1e-5)
    assert sampled.dist_velocity_mean == pytest.approx(expected, rel=1e-5)
    assert sampled.dist_velocity_mean == pytest.approx(full.dist_velocity_mean, rel=1e-5)
    assert sampled.duration == pytest.approx((399 / 80.0), abs=1e-9)


def test_bad_sample_timestamps_are_held_not_silently_scaled():
    calc = MetricsCalculator(fps=80.0)
    repeated = _ramp_landmarks(stride=3)
    repeated[6]["timestamp"] = repeated[5]["timestamp"]
    with pytest.raises(ValueError, match="timestamps must increase strictly"):
        calc.calculate_finger_tapping_metrics(repeated)

    nonfinite = _ramp_landmarks(stride=3)
    nonfinite[4]["timestamp"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        calc.calculate_finger_tapping_metrics(nonfinite)


def test_native_30fps_timing_remains_physical():
    metrics = MetricsCalculator(fps=30.0).calculate_finger_tapping_metrics(
        _ramp_landmarks(fps=30, n_frames=150, stride=1)
    )
    assert metrics.dist_velocity_mean == pytest.approx(.05, rel=1e-5)


def test_synthetic_2p5hz_peaks_remain_comparable_after_decimation():
    """A periodic trace should not lose taps solely due to source-FPS spacing."""
    import math
    calc = MetricsCalculator(fps=80.0)
    counts = []
    for stride in (1, 3):
        frames = _ramp_landmarks(stride=stride)
        for frame in frames:
            t = frame["timestamp"]
            frame["landmarks"][1]["x"] = .55 - .035 * math.sin(2 * math.pi * 2.5 * t)
        counts.append(calc.calculate_finger_tapping_metrics(frames).total_taps)
    assert counts[0] > 8
    assert counts[0] == counts[1]
