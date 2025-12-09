"""
Extract Kinematic Features from PD4T Finger Tapping Videos for ML Training
"""
import sys
import os
import csv
import time

sys.path.insert(0, 'C:/Users/YK/tulip/Hawkeye/backend')

from dataclasses import asdict
from services.mediapipe_processor import MediaPipeProcessor
from services.metrics_calculator import MetricsCalculator

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Finger tapping"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations_split/Finger tapping"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/ml_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_annotation_file(txt_path):
    """Parse annotation txt file"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_hand_subject = parts[0]
                frames = int(parts[1])
                score = int(parts[2])

                # Parse: 15-002802_r_022 -> video_id=15-002802_r, subject=022
                parts_split = video_id_hand_subject.rsplit('_', 1)
                video_id = parts_split[0]  # includes hand (l/r)
                subject = parts_split[1]

                annotations.append({
                    'video_id': video_id,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': f"{VIDEO_ROOT}/{subject}/{video_id}.mp4"
                })
    return annotations

def extract_features_from_video(video_path, processor, calculator):
    """Extract kinematic features from a single finger tapping video"""
    landmark_frames = processor.process_video(video_path)
    frames_dict = [asdict(f) for f in landmark_frames]
    metrics = calculator.calculate_finger_tapping_metrics(frames_dict)

    features = {
        # Tapping speed and count
        'tapping_speed': metrics.tapping_speed,
        'total_taps': metrics.total_taps,
        # Amplitude features
        'amplitude_mean': metrics.amplitude_mean,
        'amplitude_std': metrics.amplitude_std,
        'amplitude_decrement': metrics.amplitude_decrement,
        'first_half_amplitude': metrics.first_half_amplitude,
        'second_half_amplitude': metrics.second_half_amplitude,
        # Velocity features (MDS-UPDRS critical)
        'opening_velocity_mean': metrics.opening_velocity_mean,
        'closing_velocity_mean': metrics.closing_velocity_mean,
        'peak_velocity_mean': metrics.peak_velocity_mean,
        'velocity_decrement': metrics.velocity_decrement,
        # Rhythm features
        'rhythm_variability': metrics.rhythm_variability,
        # Events
        'hesitation_count': metrics.hesitation_count,
        'halt_count': metrics.halt_count,
        'freeze_episodes': metrics.freeze_episodes,
        # Fatigue
        'fatigue_rate': metrics.fatigue_rate,
        # Duration
        'duration': metrics.duration,
        # === TIME-SERIES FEATURES (NEW) ===
        # Temporal segments
        'velocity_first_third': metrics.velocity_first_third,
        'velocity_mid_third': metrics.velocity_mid_third,
        'velocity_last_third': metrics.velocity_last_third,
        'amplitude_first_third': metrics.amplitude_first_third,
        'amplitude_mid_third': metrics.amplitude_mid_third,
        'amplitude_last_third': metrics.amplitude_last_third,
        # Trend slopes
        'velocity_slope': metrics.velocity_slope,
        'amplitude_slope': metrics.amplitude_slope,
        'rhythm_slope': metrics.rhythm_slope,
        # Variability progression
        'variability_first_half': metrics.variability_first_half,
        'variability_second_half': metrics.variability_second_half,
        'variability_change': metrics.variability_change,
    }
    return features

def process_dataset(split_name, annotations, processor, calculator):
    """Process all videos in a dataset split"""
    results = []
    failed = []
    total = len(annotations)

    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {total} videos")
    print(f"{'='*60}")

    for i, ann in enumerate(annotations):
        video_path = ann['video_path']
        video_name = f"{ann['video_id']}_{ann['subject']}"

        print(f"[{i+1}/{total}] {video_name} (Score: {ann['score']})", end=" ")

        if not os.path.exists(video_path):
            print("- NOT FOUND")
            failed.append({'video': video_name, 'error': 'File not found'})
            continue

        try:
            start_time = time.time()
            features = extract_features_from_video(video_path, processor, calculator)
            elapsed = time.time() - start_time

            result = {
                'video_id': ann['video_id'],
                'subject': ann['subject'],
                'score': ann['score'],
                **features
            }
            results.append(result)
            print(f"- OK ({elapsed:.1f}s) speed={features['tapping_speed']:.2f}Hz")

        except Exception as e:
            print(f"- ERROR: {str(e)[:50]}")
            failed.append({'video': video_name, 'error': str(e)})

    return results, failed

def save_features(results, output_path):
    """Save features to CSV"""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} samples to {output_path}")

def main():
    print("="*60)
    print("PD4T Finger Tapping Feature Extraction")
    print("="*60)

    # Load annotations
    train_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/train.txt")
    valid_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/valid.txt")
    test_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/test.txt")

    print(f"Train: {len(train_ann)}, Valid: {len(valid_ann)}, Test: {len(test_ann)}")

    # Check score distribution
    for name, ann_list in [('Train', train_ann), ('Valid', valid_ann), ('Test', test_ann)]:
        scores = {}
        for ann in ann_list:
            s = ann['score']
            scores[s] = scores.get(s, 0) + 1
        print(f"{name} score distribution: {scores}")

    # Initialize processors
    processor = MediaPipeProcessor(mode='hand')
    calculator = MetricsCalculator(fps=30.0)

    # Process each split
    train_results, train_failed = process_dataset('train', train_ann, processor, calculator)
    valid_results, valid_failed = process_dataset('valid', valid_ann, processor, calculator)
    test_results, test_failed = process_dataset('test', test_ann, processor, calculator)

    # Save results
    save_features(train_results, f"{OUTPUT_DIR}/finger_tapping_train_features.csv")
    save_features(valid_results, f"{OUTPUT_DIR}/finger_tapping_valid_features.csv")
    save_features(test_results, f"{OUTPUT_DIR}/finger_tapping_test_features.csv")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train: {len(train_results)} success, {len(train_failed)} failed")
    print(f"Valid: {len(valid_results)} success, {len(valid_failed)} failed")
    print(f"Test: {len(test_results)} success, {len(test_failed)} failed")

if __name__ == "__main__":
    main()
