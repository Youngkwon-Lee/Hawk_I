"""
Prepare PD4T Finger Tapping data v3 - Enhanced Features

Based on prior research findings:
- PMC10674854: 80% with amplitude decay, breakpoint detection
- arXiv:2510.10121: 93% with 57 temporal/frequency features
- PMC11260436: 86% with variability features

New Features (v3): 73 raw + 10 basic clinical + 25 advanced temporal = 108 features

Advanced Features:
1. Amplitude decay (진폭 감소율) - MDS-UPDRS 핵심 지표
2. Tapping frequency (태핑 주파수)
3. Hesitation count (멈춤 횟수)
4. Variability metrics (CV of amplitude, speed, cycle)
5. Opening/closing speed separation
6. Cycle duration statistics
7. Peak detection features

Usage:
    python scripts/prepare_finger_v3.py
"""
import os
import sys
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import variation

# ============================================================
# Configuration
# ============================================================
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
ANNOTATION_DIR = f"{PD4T_ROOT}/Annotations_split/Finger tapping"
VIDEO_DIR = f"{PD4T_ROOT}/Videos/Finger tapping"
OUTPUT_DIR = "./data"

SEQUENCE_LENGTH = 150
NUM_LANDMARKS = 21
LANDMARK_DIM = 3
NUM_RAW = NUM_LANDMARKS * LANDMARK_DIM  # 63
NUM_BASIC_CLINICAL = 10
NUM_ADVANCED = 25
TOTAL_FEATURES = NUM_RAW + NUM_BASIC_CLINICAL + NUM_ADVANCED  # 63 + 10 + 25 = 98

FPS = 30  # Assumed frame rate


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


def extract_landmarks_mediapipe(video_path, max_frames=300):
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


def extract_basic_clinical_features(landmarks):
    """
    Extract basic clinical features (same as v2)
    """
    n_frames = len(landmarks)
    if n_frames < 2:
        return np.zeros((n_frames, NUM_BASIC_CLINICAL))

    thumb_pos = landmarks[:, 12:15]  # THUMB_TIP (landmark 4)
    index_pos = landmarks[:, 24:27]  # INDEX_TIP (landmark 8)
    wrist_pos = landmarks[:, 0:3]    # WRIST (landmark 0)

    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    dist_velocity = np.gradient(finger_distance)
    dist_accel = np.gradient(dist_velocity)

    thumb_diff = np.diff(thumb_pos, axis=0)
    thumb_speed = np.linalg.norm(thumb_diff, axis=1)
    thumb_speed = np.concatenate([[0], thumb_speed])

    index_diff = np.diff(index_pos, axis=0)
    index_speed = np.linalg.norm(index_diff, axis=1)
    index_speed = np.concatenate([[0], index_speed])

    combined_speed = thumb_speed + index_speed

    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 1e-6
    normalized_distance = finger_distance / hand_size

    clinical = np.stack([
        finger_distance,
        dist_velocity,
        dist_accel,
        thumb_speed,
        index_speed,
        combined_speed,
        thumb_from_wrist,
        index_from_wrist,
        normalized_distance,
        hand_size,
    ], axis=1)

    return clinical


def detect_taps(finger_distance, min_distance=0.02, prominence=0.01):
    """
    Detect individual finger taps from distance signal

    Returns:
        peaks: indices of tap peaks (finger open)
        valleys: indices of tap valleys (finger closed)
    """
    # Find peaks (maximum finger opening)
    peaks, peak_props = find_peaks(finger_distance, distance=5, prominence=prominence)

    # Find valleys (finger closed)
    valleys, valley_props = find_peaks(-finger_distance, distance=5, prominence=prominence)

    return peaks, valleys


def extract_advanced_features(landmarks, basic_clinical):
    """
    Extract advanced temporal/statistical features based on prior research

    Features (25 total):
    1-5: Amplitude features (mean, std, CV, decay_rate, decay_slope)
    6-10: Speed features (opening_speed_mean, closing_speed_mean, speed_CV, max_speed, min_speed)
    11-15: Frequency features (tap_frequency, cycle_duration_mean, cycle_duration_std, cycle_CV, num_taps)
    16-20: Hesitation features (hesitation_count, hesitation_ratio, longest_pause, pause_std, inter_tap_variability)
    21-25: Temporal features (early_amplitude, late_amplitude, amplitude_ratio, fatigue_index, rhythm_regularity)
    """
    n_frames = len(landmarks)
    if n_frames < 30:  # Need minimum frames for analysis
        return np.zeros((n_frames, NUM_ADVANCED))

    # Get finger distance signal
    finger_distance = basic_clinical[:, 0]  # First column is finger_distance
    dist_velocity = basic_clinical[:, 1]

    # Detect taps
    peaks, valleys = detect_taps(finger_distance)

    # ============================================================
    # 1-5: Amplitude Features
    # ============================================================
    if len(peaks) > 0:
        amplitudes = finger_distance[peaks]
        amp_mean = np.mean(amplitudes)
        amp_std = np.std(amplitudes)
        amp_cv = amp_std / (amp_mean + 1e-6)  # Coefficient of variation

        # Amplitude decay (linear regression slope)
        if len(peaks) >= 3:
            tap_indices = np.arange(len(peaks))
            slope, intercept = np.polyfit(tap_indices, amplitudes, 1)
            amp_decay_rate = -slope / (amp_mean + 1e-6)  # Normalized decay
            amp_decay_slope = slope
        else:
            amp_decay_rate = 0
            amp_decay_slope = 0
    else:
        amp_mean = np.mean(finger_distance)
        amp_std = np.std(finger_distance)
        amp_cv = 0
        amp_decay_rate = 0
        amp_decay_slope = 0

    # ============================================================
    # 6-10: Speed Features
    # ============================================================
    # Separate opening and closing phases
    opening_speeds = []
    closing_speeds = []

    for i in range(len(valleys) - 1):
        # Find peak between consecutive valleys
        valley_start = valleys[i]
        valley_end = valleys[i + 1]
        peaks_between = peaks[(peaks > valley_start) & (peaks < valley_end)]

        if len(peaks_between) > 0:
            peak_idx = peaks_between[0]
            # Opening: valley to peak (positive velocity)
            opening_vel = dist_velocity[valley_start:peak_idx]
            if len(opening_vel) > 0:
                opening_speeds.append(np.max(opening_vel))
            # Closing: peak to valley (negative velocity)
            closing_vel = dist_velocity[peak_idx:valley_end]
            if len(closing_vel) > 0:
                closing_speeds.append(np.abs(np.min(closing_vel)))

    if len(opening_speeds) > 0:
        opening_speed_mean = np.mean(opening_speeds)
        closing_speed_mean = np.mean(closing_speeds) if closing_speeds else 0
        all_speeds = opening_speeds + closing_speeds
        speed_cv = np.std(all_speeds) / (np.mean(all_speeds) + 1e-6)
        max_speed = np.max(all_speeds)
        min_speed = np.min(all_speeds)
    else:
        opening_speed_mean = np.mean(np.abs(dist_velocity))
        closing_speed_mean = opening_speed_mean
        speed_cv = 0
        max_speed = np.max(np.abs(dist_velocity))
        min_speed = np.min(np.abs(dist_velocity[dist_velocity != 0])) if np.any(dist_velocity != 0) else 0

    # ============================================================
    # 11-15: Frequency Features
    # ============================================================
    num_taps = len(peaks)
    duration_sec = n_frames / FPS

    tap_frequency = num_taps / duration_sec if duration_sec > 0 else 0

    # Cycle duration (time between consecutive peaks)
    if len(peaks) >= 2:
        cycle_durations = np.diff(peaks) / FPS  # in seconds
        cycle_duration_mean = np.mean(cycle_durations)
        cycle_duration_std = np.std(cycle_durations)
        cycle_cv = cycle_duration_std / (cycle_duration_mean + 1e-6)
    else:
        cycle_duration_mean = duration_sec
        cycle_duration_std = 0
        cycle_cv = 0

    # ============================================================
    # 16-20: Hesitation Features
    # ============================================================
    # Detect hesitations (pauses/freezes in movement)
    speed_threshold = np.percentile(np.abs(dist_velocity), 10)
    low_speed_frames = np.abs(dist_velocity) < speed_threshold

    # Find consecutive low-speed segments
    hesitation_count = 0
    hesitation_durations = []
    in_hesitation = False
    hesitation_start = 0

    for i, is_low in enumerate(low_speed_frames):
        if is_low and not in_hesitation:
            in_hesitation = True
            hesitation_start = i
        elif not is_low and in_hesitation:
            in_hesitation = False
            duration = (i - hesitation_start) / FPS
            if duration > 0.1:  # Minimum 100ms pause
                hesitation_count += 1
                hesitation_durations.append(duration)

    hesitation_ratio = hesitation_count / (num_taps + 1e-6)
    longest_pause = max(hesitation_durations) if hesitation_durations else 0
    pause_std = np.std(hesitation_durations) if len(hesitation_durations) > 1 else 0

    # Inter-tap interval variability
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        inter_tap_variability = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        inter_tap_variability = 0

    # ============================================================
    # 21-25: Temporal/Fatigue Features
    # ============================================================
    # Compare early vs late performance
    if len(peaks) >= 4:
        mid_point = len(peaks) // 2
        early_amplitudes = finger_distance[peaks[:mid_point]]
        late_amplitudes = finger_distance[peaks[mid_point:]]
        early_amplitude = np.mean(early_amplitudes)
        late_amplitude = np.mean(late_amplitudes)
        amplitude_ratio = late_amplitude / (early_amplitude + 1e-6)

        # Fatigue index (how much performance degraded)
        fatigue_index = 1 - amplitude_ratio if amplitude_ratio < 1 else 0
    else:
        early_amplitude = amp_mean
        late_amplitude = amp_mean
        amplitude_ratio = 1.0
        fatigue_index = 0

    # Rhythm regularity (inverse of cycle CV)
    rhythm_regularity = 1 / (1 + cycle_cv)

    # ============================================================
    # Create per-frame feature array (broadcast statistics to all frames)
    # ============================================================
    advanced = np.zeros((n_frames, NUM_ADVANCED))

    # Broadcast scalar features to all frames
    features = [
        amp_mean, amp_std, amp_cv, amp_decay_rate, amp_decay_slope,  # 1-5
        opening_speed_mean, closing_speed_mean, speed_cv, max_speed, min_speed,  # 6-10
        tap_frequency, cycle_duration_mean, cycle_duration_std, cycle_cv, num_taps / 10,  # 11-15 (normalized)
        hesitation_count / 10, hesitation_ratio, longest_pause, pause_std, inter_tap_variability,  # 16-20
        early_amplitude, late_amplitude, amplitude_ratio, fatigue_index, rhythm_regularity  # 21-25
    ]

    for i, feat in enumerate(features):
        advanced[:, i] = feat

    return advanced


def combine_all_features(landmarks, basic_clinical, advanced):
    """Combine raw 3D + basic clinical + advanced features"""
    # landmarks: (frames, 63)
    # basic_clinical: (frames, 10)
    # advanced: (frames, 25)
    return np.hstack([landmarks, basic_clinical, advanced])  # (frames, 98)


def pad_sequence(seq, target_len):
    """Pad or truncate sequence to target length"""
    if len(seq) >= target_len:
        return seq[:target_len]
    else:
        padded = np.zeros((target_len, seq.shape[1]))
        padded[:len(seq)] = seq
        return padded


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
            # Extract raw 3D landmarks
            landmarks = extract_landmarks_mediapipe(video_path)
            if len(landmarks) < 30:  # Need minimum frames
                skipped += 1
                continue

            # Extract basic clinical features
            basic_clinical = extract_basic_clinical_features(landmarks)

            # Extract advanced features
            advanced = extract_advanced_features(landmarks, basic_clinical)

            # Combine all features
            combined = combine_all_features(landmarks, basic_clinical, advanced)

            # Pad/truncate to fixed length
            padded = pad_sequence(combined, SEQUENCE_LENGTH)

            all_features.append(padded)
            all_scores.append(ann['score'])
            all_ids.append(ann['video_id'])

        except Exception as e:
            print(f"Error {ann['video_id']}: {e}")
            skipped += 1
            continue

    X = np.array(all_features)
    y = np.array(all_scores)

    print(f"  Processed: {len(X)} samples, Skipped: {skipped}")
    print(f"  Shape: {X.shape} (samples, frames, features)")

    return X, y, all_ids


def main():
    print("=" * 70)
    print("PD4T Finger Tapping v3 - Enhanced Temporal Features")
    print(f"Features per frame: {TOTAL_FEATURES}")
    print(f"  - Raw 3D: {NUM_RAW}")
    print(f"  - Basic Clinical: {NUM_BASIC_CLINICAL}")
    print(f"  - Advanced Temporal: {NUM_ADVANCED}")
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
    raw_names = []
    for i in range(21):
        landmark_names = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
                         'index_mcp', 'index_pip', 'index_dip', 'index_tip',
                         'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
                         'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
                         'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
        for coord in ['x', 'y', 'z']:
            raw_names.append(f"{landmark_names[i]}_{coord}")

    basic_clinical_names = [
        'finger_distance', 'dist_velocity', 'dist_accel',
        'thumb_speed', 'index_speed', 'combined_speed',
        'thumb_from_wrist', 'index_from_wrist',
        'normalized_distance', 'hand_size'
    ]

    advanced_names = [
        'amp_mean', 'amp_std', 'amp_cv', 'amp_decay_rate', 'amp_decay_slope',
        'opening_speed_mean', 'closing_speed_mean', 'speed_cv', 'max_speed', 'min_speed',
        'tap_frequency', 'cycle_duration_mean', 'cycle_duration_std', 'cycle_cv', 'num_taps_norm',
        'hesitation_count_norm', 'hesitation_ratio', 'longest_pause', 'pause_std', 'inter_tap_variability',
        'early_amplitude', 'late_amplitude', 'amplitude_ratio', 'fatigue_index', 'rhythm_regularity'
    ]

    feature_names = raw_names + basic_clinical_names + advanced_names

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
            'version': 'v3_enhanced_temporal',
            'features': feature_names,
            'feature_groups': {
                'raw_3d': list(range(0, NUM_RAW)),
                'basic_clinical': list(range(NUM_RAW, NUM_RAW + NUM_BASIC_CLINICAL)),
                'advanced_temporal': list(range(NUM_RAW + NUM_BASIC_CLINICAL, TOTAL_FEATURES))
            }
        }
        filename = f"{OUTPUT_DIR}/finger_{split_name}_v3.pkl"
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
    print(f"Feature shape: {X_train.shape[1:]} (seq_len, features)")

    print(f"\nAdvanced Features Added:")
    print(f"  - Amplitude decay (MDS-UPDRS key metric)")
    print(f"  - Tapping frequency and cycle duration")
    print(f"  - Hesitation/pause detection")
    print(f"  - Opening/closing speed separation")
    print(f"  - Fatigue index (early vs late performance)")
    print(f"  - Rhythm regularity")

    print(f"\nLabel distribution (Train):")
    for i in range(5):
        count = (y_train == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_train)*100:.1f}%)")


if __name__ == "__main__":
    main()
