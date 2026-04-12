"""
Extract Kinematic Features from PD4T Gait Videos for ML Training
With incremental saving and resume capability
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
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Gait"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations_split/Gait"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/ml_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    'video_id', 'subject', 'score',
    'arm_swing_amplitude_mean', 'arm_swing_amplitude_left', 'arm_swing_amplitude_right',
    'arm_swing_asymmetry', 'walking_speed', 'cadence', 'step_height_mean', 'step_count',
    'stride_length', 'stride_variability', 'swing_time_mean', 'stance_time_mean',
    'swing_stance_ratio', 'double_support_percent', 'step_length_asymmetry',
    'swing_time_asymmetry', 'duration',
    # Joint angle features (degrees)
    'trunk_flexion_mean', 'trunk_flexion_rom',
    'hip_flexion_rom_left', 'hip_flexion_rom_right', 'hip_flexion_rom_mean',
    'knee_flexion_rom_left', 'knee_flexion_rom_right', 'knee_flexion_rom_mean',
    'ankle_dorsiflexion_rom_left', 'ankle_dorsiflexion_rom_right', 'ankle_dorsiflexion_rom_mean',
    # Time-series features (NEW)
    'step_length_first_half', 'step_length_second_half', 'step_length_trend',
    'cadence_first_half', 'cadence_second_half', 'cadence_trend',
    'arm_swing_first_half', 'arm_swing_second_half', 'arm_swing_trend',
    'stride_variability_first_half', 'stride_variability_second_half', 'variability_trend',
    'step_height_first_half', 'step_height_second_half', 'step_height_trend'
]

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
                video_id_subject = parts[0]
                frames = int(parts[1])
                score = int(parts[2])

                video_parts = video_id_subject.rsplit('_', 1)
                video_id = video_parts[0]
                subject = video_parts[1]

                annotations.append({
                    'video_id': video_id,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': f"{VIDEO_ROOT}/{subject}/{video_id}.mp4"
                })
    return annotations

def load_existing_results(csv_path):
    """Load existing results to enable resume"""
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['video_id']}_{row['subject']}"
                processed.add(key)
    return processed

def extract_features_from_video(video_path, processor, calculator):
    """Extract kinematic features from a single video"""
    landmark_frames = processor.process_video(video_path)
    frames_dict = [asdict(f) for f in landmark_frames]
    metrics = calculator.calculate_gait_metrics(frames_dict)

    features = {
        # Arm swing features
        'arm_swing_amplitude_mean': metrics.arm_swing_amplitude_mean,
        'arm_swing_amplitude_left': metrics.arm_swing_amplitude_left,
        'arm_swing_amplitude_right': metrics.arm_swing_amplitude_right,
        'arm_swing_asymmetry': metrics.arm_swing_asymmetry,
        # Speed and cadence
        'walking_speed': metrics.walking_speed,
        'cadence': metrics.cadence,
        # Step features
        'step_height_mean': metrics.step_height_mean,
        'step_count': metrics.step_count,
        # Stride features
        'stride_length': metrics.stride_length,
        'stride_variability': metrics.stride_variability,
        # Gait phases
        'swing_time_mean': metrics.swing_time_mean,
        'stance_time_mean': metrics.stance_time_mean,
        'swing_stance_ratio': metrics.swing_stance_ratio,
        'double_support_percent': metrics.double_support_percent,
        # Asymmetry
        'step_length_asymmetry': metrics.step_length_asymmetry,
        'swing_time_asymmetry': metrics.swing_time_asymmetry,
        # Duration
        'duration': metrics.duration,
        # Joint angle features (degrees)
        'trunk_flexion_mean': metrics.trunk_flexion_mean,
        'trunk_flexion_rom': metrics.trunk_flexion_rom,
        'hip_flexion_rom_left': metrics.hip_flexion_rom_left,
        'hip_flexion_rom_right': metrics.hip_flexion_rom_right,
        'hip_flexion_rom_mean': metrics.hip_flexion_rom_mean,
        'knee_flexion_rom_left': metrics.knee_flexion_rom_left,
        'knee_flexion_rom_right': metrics.knee_flexion_rom_right,
        'knee_flexion_rom_mean': metrics.knee_flexion_rom_mean,
        'ankle_dorsiflexion_rom_left': metrics.ankle_dorsiflexion_rom_left,
        'ankle_dorsiflexion_rom_right': metrics.ankle_dorsiflexion_rom_right,
        'ankle_dorsiflexion_rom_mean': metrics.ankle_dorsiflexion_rom_mean,
        # Time-series features (NEW)
        'step_length_first_half': metrics.step_length_first_half,
        'step_length_second_half': metrics.step_length_second_half,
        'step_length_trend': metrics.step_length_trend,
        'cadence_first_half': metrics.cadence_first_half,
        'cadence_second_half': metrics.cadence_second_half,
        'cadence_trend': metrics.cadence_trend,
        'arm_swing_first_half': metrics.arm_swing_first_half,
        'arm_swing_second_half': metrics.arm_swing_second_half,
        'arm_swing_trend': metrics.arm_swing_trend,
        'stride_variability_first_half': metrics.stride_variability_first_half,
        'stride_variability_second_half': metrics.stride_variability_second_half,
        'variability_trend': metrics.variability_trend,
        'step_height_first_half': metrics.step_height_first_half,
        'step_height_second_half': metrics.step_height_second_half,
        'step_height_trend': metrics.step_height_trend,
    }
    return features

def process_dataset_incremental(split_name, annotations, processor, calculator, output_path):
    """Process videos with incremental saving"""
    # Load already processed
    processed = load_existing_results(output_path)
    print(f"Already processed: {len(processed)} videos")

    # Check if file exists, if not create with header
    file_exists = os.path.exists(output_path) and len(processed) > 0

    success_count = len(processed)
    failed = []
    total = len(annotations)

    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {total} videos ({len(processed)} already done)")
    print(f"{'='*60}")

    with open(output_path, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, ann in enumerate(annotations):
            video_path = ann['video_path']
            video_name = f"{ann['video_id']}_{ann['subject']}"

            # Skip if already processed
            if video_name in processed:
                continue

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

                # Write immediately
                writer.writerow(result)
                f.flush()  # Ensure written to disk

                success_count += 1
                print(f"- OK ({elapsed:.1f}s) arm={features['arm_swing_amplitude_mean']*100:.1f}cm")

            except Exception as e:
                print(f"- ERROR: {str(e)[:50]}")
                failed.append({'video': video_name, 'error': str(e)})

    return success_count, failed

def main():
    print("="*60)
    print("PD4T Gait Feature Extraction (Incremental)")
    print("="*60)

    # Load annotations
    train_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/train.txt")
    valid_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/valid.txt")
    test_ann = parse_annotation_file(f"{ANNOTATION_ROOT}/test.txt")

    print(f"Train: {len(train_ann)}, Valid: {len(valid_ann)}, Test: {len(test_ann)}")

    # Initialize processors
    processor = MediaPipeProcessor(mode='pose')
    calculator = MetricsCalculator(fps=30.0)

    # Process each split with incremental saving
    train_success, train_failed = process_dataset_incremental(
        'train', train_ann, processor, calculator,
        f"{OUTPUT_DIR}/gait_train_features.csv"
    )
    valid_success, valid_failed = process_dataset_incremental(
        'valid', valid_ann, processor, calculator,
        f"{OUTPUT_DIR}/gait_valid_features.csv"
    )
    test_success, test_failed = process_dataset_incremental(
        'test', test_ann, processor, calculator,
        f"{OUTPUT_DIR}/gait_test_features.csv"
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train: {train_success} success, {len(train_failed)} failed")
    print(f"Valid: {valid_success} success, {len(valid_failed)} failed")
    print(f"Test: {test_success} success, {len(test_failed)} failed")

if __name__ == "__main__":
    main()
