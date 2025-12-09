"""
Test Gait Cycle Analyzer
Processes a sample gait video and runs detailed cycle analysis
"""
import sys
import os
import json

sys.path.insert(0, 'C:/Users/YK/tulip/Hawkeye/backend')

from services.mediapipe_processor import MediaPipeProcessor
from services.gait_cycle_analyzer import GaitCycleAnalyzer
from dataclasses import asdict

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Gait"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations_split/Gait"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_sample_video():
    """Find a sample gait video from the dataset"""
    # Read test annotations
    test_txt = f"{ANNOTATION_ROOT}/test.txt"
    if not os.path.exists(test_txt):
        print(f"Annotation file not found: {test_txt}")
        return None, None, None

    with open(test_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_subject = parts[0]
                score = int(parts[2])

                # Try to find the video file
                video_parts = video_id_subject.rsplit('_', 1)
                video_id = video_parts[0]
                subject = video_parts[1] if len(video_parts) > 1 else 'unknown'

                # Try common extensions
                for ext in ['.mp4', '.avi', '.MP4', '.AVI']:
                    video_path = f"{VIDEO_ROOT}/{video_id}{ext}"
                    if os.path.exists(video_path):
                        return video_path, video_id, score

    return None, None, None


def process_and_analyze(video_path: str, video_id: str, score: int):
    """Process video and run gait cycle analysis"""
    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"UPDRS Score: {score}")
    print(f"Path: {video_path}")
    print('='*60)

    # Initialize processor (pose mode for gait analysis)
    processor = MediaPipeProcessor(mode="pose")

    # Process video
    print("\n[1/3] Extracting pose landmarks...")
    landmark_frames_raw = processor.process_video(video_path)

    if not landmark_frames_raw or len(landmark_frames_raw) < 30:
        print(f"Error: Insufficient frames extracted ({len(landmark_frames_raw) if landmark_frames_raw else 0})")
        return

    print(f"  Extracted {len(landmark_frames_raw)} frames")

    # Convert LandmarkFrame objects to dictionaries
    landmark_frames = []
    for frame in landmark_frames_raw:
        frame_dict = {
            'frame_number': frame.frame_number,
            'timestamp': frame.timestamp,
            'keypoints': frame.landmarks,
        }
        if frame.world_landmarks:
            frame_dict['world_keypoints'] = frame.world_landmarks
        landmark_frames.append(frame_dict)

    # Save skeleton data
    skeleton_path = f"{OUTPUT_DIR}/{video_id}_skeleton.json"
    with open(skeleton_path, 'w') as f:
        json.dump(landmark_frames, f)
    print(f"  Saved skeleton to: {skeleton_path}")

    # Run gait cycle analysis
    print("\n[2/3] Analyzing gait cycles...")
    analyzer = GaitCycleAnalyzer(fps=30.0)

    try:
        analysis = analyzer.analyze(landmark_frames)
        result = analyzer.to_dict(analysis)

        # Save analysis results
        analysis_path = f"{OUTPUT_DIR}/{video_id}_gait_cycle_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved analysis to: {analysis_path}")

        # Print results
        print("\n[3/3] Results:")
        print_analysis_results(result, score)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def print_analysis_results(result: dict, updrs_score: int):
    """Print formatted analysis results"""
    summary = result['summary']
    timing = result['timing']
    phase = result['phase_distribution']
    asym = result['asymmetry_percent']
    var = result['variability_cv']

    print("\n" + "="*60)
    print("GAIT CYCLE ANALYSIS RESULTS")
    print("="*60)

    print(f"\n--- Summary ---")
    print(f"Total Gait Cycles: {summary['total_cycles']} (L:{summary['num_cycles_left']}, R:{summary['num_cycles_right']})")
    print(f"Analysis Duration: {summary['analysis_duration_sec']:.1f} sec")
    print(f"UPDRS Gait Score: {updrs_score}")

    print(f"\n--- Timing ---")
    print(f"Cycle Time: {timing['cycle_time_mean_sec']:.3f} ± {timing['cycle_time_std_sec']:.3f} sec")
    print(f"Cadence: {60/timing['cycle_time_mean_sec']:.1f} steps/min" if timing['cycle_time_mean_sec'] > 0 else "Cadence: N/A")
    print(f"Cycle Variability (CV): {timing['cycle_time_cv_percent']:.1f}%")

    print(f"\n--- Phase Distribution ---")
    print(f"Stance Phase: {phase['stance_percent_mean']:.1f} ± {phase['stance_percent_std']:.1f}%")
    print(f"Swing Phase: {phase['swing_percent_mean']:.1f} ± {phase['swing_percent_std']:.1f}%")
    print(f"Double Support: {phase['double_support_percent']:.1f}%")
    print(f"Single Support: {phase['single_support_percent']:.1f}%")

    # Normal values comparison
    print(f"\n--- Comparison with Normal Values ---")
    normal_stance = 60.0  # Normal stance ~60%
    normal_swing = 40.0   # Normal swing ~40%
    normal_ds = 20.0      # Normal double support ~20%

    stance_diff = phase['stance_percent_mean'] - normal_stance
    swing_diff = phase['swing_percent_mean'] - normal_swing
    ds_diff = phase['double_support_percent'] - normal_ds

    print(f"Stance: {'⬆️ +' if stance_diff > 0 else '⬇️ '}{stance_diff:.1f}% from normal (60%)")
    print(f"Swing:  {'⬆️ +' if swing_diff > 0 else '⬇️ '}{swing_diff:.1f}% from normal (40%)")
    print(f"Double Support: {'⬆️ +' if ds_diff > 0 else '⬇️ '}{ds_diff:.1f}% from normal (20%)")

    print(f"\n--- Asymmetry ---")
    print(f"Stance Time Asymmetry: {asym['stance_time']:.1f}%")
    print(f"Swing Time Asymmetry: {asym['swing_time']:.1f}%")
    print(f"Step Length Asymmetry: {asym['step_length']:.1f}%")

    # Asymmetry interpretation
    max_asym = max(asym['stance_time'], asym['swing_time'], asym['step_length'])
    if max_asym > 15:
        print("  ⚠️  Significant asymmetry detected (>15%)")
    elif max_asym > 10:
        print("  ℹ️  Moderate asymmetry (10-15%)")
    else:
        print("  ✓  Symmetry within normal limits (<10%)")

    print(f"\n--- Variability (Cycle-to-Cycle) ---")
    print(f"Step Length CV: {var['step_length']:.1f}%")
    print(f"Step Time CV: {var['step_time']:.1f}%")
    print(f"Stride Time CV: {var['stride_time']:.1f}%")

    # Variability interpretation (CV > 10% is concerning)
    if var['stride_time'] > 10:
        print("  ⚠️  High stride variability (>10%) - may indicate gait instability")
    elif var['stride_time'] > 5:
        print("  ℹ️  Moderate variability (5-10%)")
    else:
        print("  ✓  Low variability (<5%) - stable gait")

    # Sub-phase variability
    sub_var = result['sub_phase_variability_cv']
    print(f"\n--- Sub-Phase Variability (CV %) ---")
    print(f"Loading Response: {sub_var['loading_response']:.1f}%")
    print(f"Mid Stance: {sub_var['mid_stance']:.1f}%")
    print(f"Terminal Stance: {sub_var['terminal_stance']:.1f}%")
    print(f"Pre-Swing: {sub_var['pre_swing']:.1f}%")
    print(f"Initial Swing: {sub_var['initial_swing']:.1f}%")
    print(f"Mid Swing: {sub_var['mid_swing']:.1f}%")
    print(f"Terminal Swing: {sub_var['terminal_swing']:.1f}%")

    # Sample cycle details
    if result['cycles']['left']:
        c = result['cycles']['left'][0]
        print(f"\n--- Sample Left Cycle (#{c['cycle_number']}) ---")
        print(f"Duration: {c['durations_sec']['total']:.3f} sec")
        print(f"Stance/Swing: {c['phase_percent']['stance']:.1f}% / {c['phase_percent']['swing']:.1f}%")
        print(f"Step Length: {c['spatial_meters']['step_length']:.3f} m")
        print(f"Step Height: {c['spatial_meters']['step_height']:.3f} m")
        print(f"Arm Swing: {c['spatial_meters']['arm_swing']:.3f} m")

    # Clinical interpretation based on UPDRS score
    print(f"\n--- Clinical Interpretation ---")
    print(f"UPDRS Score {updrs_score} Characteristics:")
    if updrs_score == 0:
        print("  Normal gait expected")
    elif updrs_score == 1:
        print("  Mild slowness or reduced arm swing, may not be visible without careful observation")
    elif updrs_score == 2:
        print("  Moderately impaired gait, visible abnormalities")
    elif updrs_score == 3:
        print("  Severely impaired gait, may need assistance")
    elif updrs_score == 4:
        print("  Cannot walk or requires walker/wheelchair")

    print("="*60)


def main():
    print("="*60)
    print("GAIT CYCLE ANALYZER TEST")
    print("="*60)

    # Check for command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        video_id = os.path.basename(video_path).rsplit('.', 1)[0]
        score = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    else:
        # Find a sample video
        print("\nSearching for sample video...")
        video_path, video_id, score = find_sample_video()

        if not video_path:
            print("No sample video found. Please provide video path as argument.")
            print("Usage: python test_gait_cycle_analyzer.py <video_path> [updrs_score]")
            return

    # Process and analyze
    process_and_analyze(video_path, video_id, score)


if __name__ == "__main__":
    main()
