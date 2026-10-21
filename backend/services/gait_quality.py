"""Independent, camera-relative gait periodicity QA (not a clinical cadence estimate)."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def periodicity_cadence_reference(frames, fps, min_visibility=0.5):
    """Return a rough steps/min cross-check from repeated ankle separation peaks.

    This is deliberately independent of the heel-strike detector. One full
    ankle-separation oscillation corresponds approximately to two steps.
    Return None when evidence is insufficient; never substitute this value
    for the primary clinical metric without validation against ground truth.
    """
    if not fps or fps <= 0 or not frames or len(frames) < max(30, int(fps * 4)):
        return None
    samples = []
    for frame in frames:
        points = frame.get("landmarks", frame.get("keypoints", []))
        points = {p.get("id"): p for p in points}
        left, right = points.get(27), points.get(28)
        if not left or not right or min(left.get("visibility", 0), right.get("visibility", 0)) < min_visibility:
            samples.append(np.nan)
        else:
            samples.append(float(left["x"]) - float(right["x"]))
    series = np.asarray(samples, dtype=float)
    if np.mean(np.isfinite(series)) < 0.8:
        return None
    if not np.all(np.isfinite(series)):
        good = np.flatnonzero(np.isfinite(series))
        series = np.interp(np.arange(len(series)), good, series[good])
    window = min(15, len(series) if len(series) % 2 else len(series) - 1)
    smooth = savgol_filter(series, window, 2)
    amplitude = float(np.ptp(smooth))
    if amplitude < 0.025:
        return None
    peaks, _ = find_peaks(smooth, distance=max(1, int(fps * 0.55)), prominence=0.18 * amplitude)
    if len(peaks) < 5:
        return None
    periods = np.diff(peaks) / fps
    median_period = float(np.median(periods))
    if not (0.65 <= median_period <= 2.5):
        return None
    # Reject irregular patterns; not a substitute for verified heel strikes.
    if float(np.std(periods) / np.mean(periods)) > 0.3:
        return None
    return round(120.0 / median_period, 1)
