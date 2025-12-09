"""
Prepare PD4T Finger Tapping data v4 - Clinical Kinematic Features

Based on PMC10674854 (80% accuracy) and PMC11260436 (86% accuracy):

NEW in v4 (vs v3):
1. Breakpoint Analysis - Piecewise linear regression for amplitude decay
   - breakpoint_tap: Which tap number fatigue starts
   - slope1: Pre-breakpoint slope (should be ~0 in normal)
   - slope2: Post-breakpoint slope (steeper = more fatigue)

2. Intra-tap Variability - Within-cycle consistency
   - opening_duration_cv: Variability in opening phase duration
   - closing_duration_cv: Variability in closing phase duration
   - peak_timing_cv: Variability in peak timing within cycles

3. Sequence Effects - Performance changes across sequence
   - first_5_taps_mean: Initial performance
   - last_5_taps_mean: Final performance
   - sequence_trend: Linear trend across all taps

4. Exponential Amplitude Decay
   - exp_decay_rate: Exponential decay constant
   - decay_r_squared: Goodness of fit for decay model

Feature structure:
- Raw 3D: 63 (21 landmarks × 3 coords) - REMOVED (not needed for clinical scoring)
- Clinical Kinematic: 35 (focused, research-validated features)
Total: 35 features per sample (aggregated, not per-frame)

Usage:
    python scripts/prepare_finger_v4.py
"""
import os
import sys
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import variation
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
ANNOTATION_DIR = f"{PD4T_ROOT}/Annotations_split/Finger tapping"
VIDEO_DIR = f"{PD4T_ROOT}/Videos/Finger tapping"
OUTPUT_DIR = "./data"

NUM_LANDMARKS = 21
LANDMARK_DIM = 3
FPS = 30

# Feature configuration
NUM_CLINICAL_FEATURES = 35


def parse_annotation_txt(txt_path):
    """Parse Annotations_split txt file"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_full = parts[0]
                frames = int(parts[1])
                score = int(parts[2])

                parts_split = video_id_full.rsplit('_', 2)
                base_id = parts_split[0]
                hand = parts_split[1]
                subject = parts_split[2]

                video_path = f"{VIDEO_DIR}/{subject}/{base_id}_{hand}.mp4"

                annotations.append({
                    'video_id': video_id_full,
                    'base_id': base_id,
                    'hand': hand,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': video_path
                })
    return annotations


def extract_landmarks_mediapipe(video_path, max_frames=500):
    """Extract hand landmarks using MediaPipe"""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    hands.close()

    return np.array(landmarks_list)


def compute_finger_distance(landmarks):
    """Compute thumb-index finger distance over time"""
    # THUMB_TIP = landmark 4 (indices 12:15)
    # INDEX_TIP = landmark 8 (indices 24:27)
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]

    distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    return distance


def detect_taps(finger_distance, min_distance=5):
    """
    Detect individual finger taps from distance signal

    Returns:
        peaks: indices of tap peaks (finger open - maximum aperture)
        valleys: indices of tap valleys (finger closed)
    """
    # Smooth the signal
    if len(finger_distance) > 11:
        smoothed = savgol_filter(finger_distance, window_length=11, polyorder=3)
    else:
        smoothed = finger_distance

    # Adaptive prominence based on signal range
    signal_range = np.max(smoothed) - np.min(smoothed)
    prominence = signal_range * 0.1

    # Find peaks (maximum finger opening)
    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)

    # Find valleys (finger closed)
    valleys, _ = find_peaks(-smoothed, distance=min_distance, prominence=prominence)

    return peaks, valleys, smoothed


def piecewise_linear(x, x0, y0, k1, k2):
    """Piecewise linear function with one breakpoint"""
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))


def find_breakpoint(amplitudes):
    """
    Find breakpoint in amplitude sequence using piecewise linear regression

    Returns:
        breakpoint_idx: Tap number where fatigue starts (0 to n_taps)
        slope1: Pre-breakpoint slope
        slope2: Post-breakpoint slope
        breakpoint_normalized: Breakpoint as fraction of total taps
    """
    n_taps = len(amplitudes)
    if n_taps < 5:
        return n_taps // 2, 0, 0, 0.5

    x = np.arange(n_taps)

    try:
        # Initial guess: breakpoint at middle, zero slopes
        p0 = [n_taps // 2, np.mean(amplitudes), 0, 0]
        bounds = ([2, 0, -np.inf, -np.inf], [n_taps - 2, np.inf, np.inf, np.inf])

        popt, _ = curve_fit(piecewise_linear, x, amplitudes, p0=p0, bounds=bounds, maxfev=1000)
        breakpoint_idx = int(popt[0])
        slope1 = popt[2]
        slope2 = popt[3]
        breakpoint_normalized = breakpoint_idx / n_taps

    except Exception:
        # Fallback: simple split
        breakpoint_idx = n_taps // 2
        slope1 = np.polyfit(x[:breakpoint_idx], amplitudes[:breakpoint_idx], 1)[0] if breakpoint_idx > 1 else 0
        slope2 = np.polyfit(x[breakpoint_idx:], amplitudes[breakpoint_idx:], 1)[0] if n_taps - breakpoint_idx > 1 else 0
        breakpoint_normalized = 0.5

    return breakpoint_idx, slope1, slope2, breakpoint_normalized


def exponential_decay(x, a, b, c):
    """Exponential decay function: a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def fit_exponential_decay(amplitudes):
    """
    Fit exponential decay model to amplitude sequence

    Returns:
        decay_rate: Exponential decay constant (higher = faster decay)
        r_squared: Goodness of fit
    """
    n_taps = len(amplitudes)
    if n_taps < 5:
        return 0, 0

    x = np.arange(n_taps)

    try:
        # Initial guess
        a0 = amplitudes[0] - amplitudes[-1]
        b0 = 0.1
        c0 = amplitudes[-1]

        popt, _ = curve_fit(exponential_decay, x, amplitudes,
                           p0=[a0, b0, c0], maxfev=1000,
                           bounds=([0, 0, 0], [np.inf, 1, np.inf]))

        decay_rate = popt[1]

        # Compute R-squared
        y_pred = exponential_decay(x, *popt)
        ss_res = np.sum((amplitudes - y_pred) ** 2)
        ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    except Exception:
        decay_rate = 0
        r_squared = 0

    return decay_rate, max(0, r_squared)


def extract_clinical_features(landmarks):
    """
    Extract 35 clinical kinematic features based on prior research

    Feature groups:
    1-7: Amplitude features (mean, std, CV, decay_linear, breakpoint, slope1, slope2)
    8-14: Speed features (opening_mean, closing_mean, speed_cv, max, min, opening_cv, closing_cv)
    15-21: Frequency features (tap_freq, cycle_mean, cycle_std, cycle_cv, num_taps, duration, taps_per_sec)
    22-28: Hesitation features (count, ratio, longest, mean, intertap_cv, freeze_count, movement_time_ratio)
    29-35: Temporal features (first5_amp, last5_amp, amp_ratio, exp_decay, exp_r2, sequence_trend, regularity)
    """
    n_frames = len(landmarks)
    features = np.zeros(NUM_CLINICAL_FEATURES)

    if n_frames < 30:
        return features

    # Compute finger distance
    finger_distance = compute_finger_distance(landmarks)

    # Detect taps
    peaks, valleys, smoothed = detect_taps(finger_distance)

    n_taps = len(peaks)
    duration_sec = n_frames / FPS

    if n_taps < 3:
        # Not enough taps for meaningful analysis
        features[4] = n_taps  # num_taps
        features[5] = duration_sec  # duration
        return features

    # Get amplitudes at peaks
    amplitudes = smoothed[peaks]

    # ============================================================
    # 1-7: Amplitude Features
    # ============================================================
    amp_mean = np.mean(amplitudes)
    amp_std = np.std(amplitudes)
    amp_cv = amp_std / (amp_mean + 1e-6)

    # Linear decay
    tap_indices = np.arange(len(amplitudes))
    decay_linear = np.polyfit(tap_indices, amplitudes, 1)[0]

    # Breakpoint analysis
    breakpoint_idx, slope1, slope2, breakpoint_norm = find_breakpoint(amplitudes)

    features[0] = amp_mean
    features[1] = amp_std
    features[2] = amp_cv
    features[3] = decay_linear
    features[4] = breakpoint_norm  # Normalized breakpoint position
    features[5] = slope1
    features[6] = slope2

    # ============================================================
    # 8-14: Speed Features
    # ============================================================
    velocity = np.gradient(smoothed)

    opening_speeds = []
    closing_speeds = []
    opening_durations = []
    closing_durations = []

    for i in range(len(valleys) - 1):
        valley_start = valleys[i]
        valley_end = valleys[i + 1]
        peaks_between = peaks[(peaks > valley_start) & (peaks < valley_end)]

        if len(peaks_between) > 0:
            peak_idx = peaks_between[0]

            # Opening phase: valley to peak
            opening_vel = velocity[valley_start:peak_idx]
            if len(opening_vel) > 0:
                opening_speeds.append(np.max(opening_vel))
                opening_durations.append((peak_idx - valley_start) / FPS)

            # Closing phase: peak to valley
            closing_vel = velocity[peak_idx:valley_end]
            if len(closing_vel) > 0:
                closing_speeds.append(np.abs(np.min(closing_vel)))
                closing_durations.append((valley_end - peak_idx) / FPS)

    if len(opening_speeds) > 0:
        opening_speed_mean = np.mean(opening_speeds)
        closing_speed_mean = np.mean(closing_speeds) if closing_speeds else 0
        all_speeds = opening_speeds + closing_speeds
        speed_cv = np.std(all_speeds) / (np.mean(all_speeds) + 1e-6)
        max_speed = np.max(all_speeds)
        min_speed = np.min(all_speeds)
        opening_cv = np.std(opening_speeds) / (np.mean(opening_speeds) + 1e-6)
        closing_cv = np.std(closing_speeds) / (np.mean(closing_speeds) + 1e-6) if len(closing_speeds) > 1 else 0
    else:
        opening_speed_mean = np.mean(np.abs(velocity))
        closing_speed_mean = opening_speed_mean
        speed_cv = 0
        max_speed = np.max(np.abs(velocity))
        min_speed = 0
        opening_cv = 0
        closing_cv = 0

    features[7] = opening_speed_mean
    features[8] = closing_speed_mean
    features[9] = speed_cv
    features[10] = max_speed
    features[11] = min_speed
    features[12] = opening_cv  # Intra-tap variability - opening
    features[13] = closing_cv  # Intra-tap variability - closing

    # ============================================================
    # 15-21: Frequency Features
    # ============================================================
    tap_frequency = n_taps / duration_sec

    if len(peaks) >= 2:
        cycle_durations = np.diff(peaks) / FPS
        cycle_mean = np.mean(cycle_durations)
        cycle_std = np.std(cycle_durations)
        cycle_cv = cycle_std / (cycle_mean + 1e-6)
    else:
        cycle_mean = duration_sec
        cycle_std = 0
        cycle_cv = 0

    features[14] = tap_frequency
    features[15] = cycle_mean
    features[16] = cycle_std
    features[17] = cycle_cv
    features[18] = n_taps
    features[19] = duration_sec
    features[20] = n_taps / duration_sec  # Taps per second (redundant but explicit)

    # ============================================================
    # 22-28: Hesitation Features
    # ============================================================
    speed_abs = np.abs(velocity)
    speed_threshold = np.percentile(speed_abs, 10)
    low_speed_mask = speed_abs < speed_threshold

    # Find hesitation episodes
    hesitation_count = 0
    hesitation_durations = []
    in_hesitation = False
    hesitation_start = 0

    for i, is_low in enumerate(low_speed_mask):
        if is_low and not in_hesitation:
            in_hesitation = True
            hesitation_start = i
        elif not is_low and in_hesitation:
            in_hesitation = False
            duration = (i - hesitation_start) / FPS
            if duration > 0.1:  # >100ms
                hesitation_count += 1
                hesitation_durations.append(duration)

    hesitation_ratio = hesitation_count / (n_taps + 1e-6)
    longest_pause = max(hesitation_durations) if hesitation_durations else 0
    mean_pause = np.mean(hesitation_durations) if hesitation_durations else 0

    # Inter-tap interval variability
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        intertap_cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        intertap_cv = 0

    # Freeze episodes (long pauses > 0.5s)
    freeze_count = sum(1 for d in hesitation_durations if d > 0.5)

    # Movement time ratio
    movement_frames = np.sum(speed_abs > speed_threshold)
    movement_time_ratio = movement_frames / n_frames

    features[21] = hesitation_count
    features[22] = hesitation_ratio
    features[23] = longest_pause
    features[24] = mean_pause
    features[25] = intertap_cv
    features[26] = freeze_count
    features[27] = movement_time_ratio

    # ============================================================
    # 29-35: Temporal/Fatigue Features
    # ============================================================
    # First and last 5 taps comparison
    if n_taps >= 10:
        first5_amp = np.mean(amplitudes[:5])
        last5_amp = np.mean(amplitudes[-5:])
    elif n_taps >= 4:
        mid = n_taps // 2
        first5_amp = np.mean(amplitudes[:mid])
        last5_amp = np.mean(amplitudes[mid:])
    else:
        first5_amp = amp_mean
        last5_amp = amp_mean

    amp_ratio = last5_amp / (first5_amp + 1e-6)

    # Exponential decay fitting
    exp_decay_rate, exp_r2 = fit_exponential_decay(amplitudes)

    # Sequence trend (linear regression slope normalized)
    sequence_trend = decay_linear / (amp_mean + 1e-6)

    # Rhythm regularity (inverse of cycle CV)
    rhythm_regularity = 1 / (1 + cycle_cv)

    features[28] = first5_amp
    features[29] = last5_amp
    features[30] = amp_ratio
    features[31] = exp_decay_rate
    features[32] = exp_r2
    features[33] = sequence_trend
    features[34] = rhythm_regularity

    return features


def process_split(annotations, split_name):
    """Process all videos in a split"""
    print(f"\nProcessing {split_name}: {len(annotations)} videos")

    all_features = []
    all_scores = []
    all_ids = []
    skipped = 0

    for ann in tqdm(annotations, desc=split_name):
        video_path = ann['video_path']

        if not os.path.exists(video_path):
            skipped += 1
            continue

        try:
            # Extract landmarks
            landmarks = extract_landmarks_mediapipe(video_path)
            if len(landmarks) < 30:
                skipped += 1
                continue

            # Extract clinical features (35 features per sample)
            features = extract_clinical_features(landmarks)

            all_features.append(features)
            all_scores.append(ann['score'])
            all_ids.append(ann['video_id'])

        except Exception as e:
            print(f"Error {ann['video_id']}: {e}")
            skipped += 1
            continue

    X = np.array(all_features)
    y = np.array(all_scores)

    print(f"  Processed: {len(X)} samples, Skipped: {skipped}")
    print(f"  Shape: {X.shape} (samples, features)")

    return X, y, all_ids


def main():
    print("=" * 70)
    print("PD4T Finger Tapping v4 - Clinical Kinematic Features")
    print("Based on PMC10674854 (80%) and PMC11260436 (86%)")
    print(f"Total Features: {NUM_CLINICAL_FEATURES}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load annotations
    print("\nLoading annotations...")
    train_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/train.txt")
    valid_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/valid.txt")
    test_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/test.txt")

    print(f"Train: {len(train_ann)}, Valid: {len(valid_ann)}, Test: {len(test_ann)}")

    # Process each split
    X_train, y_train, ids_train = process_split(train_ann, "train")
    X_valid, y_valid, ids_valid = process_split(valid_ann, "valid")
    X_test, y_test, ids_test = process_split(test_ann, "test")

    # Feature names
    feature_names = [
        # 1-7: Amplitude
        'amp_mean', 'amp_std', 'amp_cv', 'amp_decay_linear',
        'breakpoint_norm', 'slope1', 'slope2',
        # 8-14: Speed
        'opening_speed_mean', 'closing_speed_mean', 'speed_cv',
        'max_speed', 'min_speed', 'opening_cv', 'closing_cv',
        # 15-21: Frequency
        'tap_frequency', 'cycle_mean', 'cycle_std', 'cycle_cv',
        'num_taps', 'duration', 'taps_per_sec',
        # 22-28: Hesitation
        'hesitation_count', 'hesitation_ratio', 'longest_pause',
        'mean_pause', 'intertap_cv', 'freeze_count', 'movement_time_ratio',
        # 29-35: Temporal
        'first5_amp', 'last5_amp', 'amp_ratio',
        'exp_decay_rate', 'exp_r2', 'sequence_trend', 'rhythm_regularity'
    ]

    # Save each split
    print("\nSaving data...")

    for split_name, X, y, ids in [
        ('train', X_train, y_train, ids_train),
        ('valid', X_valid, y_valid, ids_valid),
        ('test', X_test, y_test, ids_test)
    ]:
        data = {
            'X': X,
            'y': y,
            'ids': ids,
            'task': 'finger_tapping',
            'version': 'v4_clinical_kinematic',
            'features': feature_names,
            'feature_groups': {
                'amplitude': list(range(0, 7)),
                'speed': list(range(7, 14)),
                'frequency': list(range(14, 21)),
                'hesitation': list(range(21, 28)),
                'temporal': list(range(28, 35))
            }
        }
        filename = f"{OUTPUT_DIR}/finger_{split_name}_v4.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved: {filename}")

    # Summary
    print(f"\n{'='*70}")
    print("Extraction Complete!")
    print(f"{'='*70}")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Valid: {X_valid.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    print(f"\nKey Features Added (vs v3):")
    print(f"  + Breakpoint analysis (piecewise linear regression)")
    print(f"  + Slope1/Slope2 (pre/post breakpoint decay)")
    print(f"  + Intra-tap variability (opening_cv, closing_cv)")
    print(f"  + Exponential decay fitting")
    print(f"  + First5/Last5 amplitude comparison")
    print(f"  + Movement time ratio")

    print(f"\nLabel distribution (Train):")
    for i in range(5):
        count = (y_train == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_train)*100:.1f}%)")

    # Feature statistics
    print(f"\nFeature Statistics (Train):")
    for i, name in enumerate(feature_names):
        values = X_train[:, i]
        print(f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")


if __name__ == "__main__":
    main()
