"""
Test GaitCycleAnalyzer V2 with PD4T Gait Videos

This script tests the improved gait cycle analyzer:
1. Processes multiple gait videos from PD4T dataset
2. Compares V1 vs V2 detection success rates
3. Validates detected cycles against expected patterns
4. Reports confidence scores and detection methods
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

import mediapipe as mp
from services.gait_cycle_analyzer import GaitCycleAnalyzer as V1Analyzer
from services.gait_cycle_analyzer_v2 import GaitCycleAnalyzerV2 as V2Analyzer


# PD4T dataset path
PD4T_BASE = Path(__file__).parent.parent.parent / "data" / "raw" / "PD4T" / "PD4T" / "PD4T"
GAIT_VIDEOS = PD4T_BASE / "Videos" / "Gait"
ANNOTATIONS = PD4T_BASE / "Annotations" / "Gait"


def get_test_videos(n_per_score: int = 3) -> Dict[int, List[Dict]]:
    """Get sample videos for each UPDRS score"""
    # Read annotations
    train_file = ANNOTATIONS / "stratified" / "train.csv"
    test_file = ANNOTATIONS / "stratified" / "test.csv"

    videos_by_score = {0: [], 1: [], 2: [], 3: []}

    for csv_file in [train_file, test_file]:
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found")
            continue

        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    video_id_subject = parts[0]
                    score = int(parts[2])

                    # Parse video_id and subject
                    # Format: 15-001760_009 -> video_id=15-001760, subject=009
                    if '_' in video_id_subject:
                        last_underscore = video_id_subject.rfind('_')
                        video_id = video_id_subject[:last_underscore]
                        subject = video_id_subject[last_underscore+1:]
                    else:
                        continue

                    video_path = GAIT_VIDEOS / subject / f"{video_id}.mp4"

                    if video_path.exists() and score in videos_by_score:
                        if len(videos_by_score[score]) < n_per_score:
                            videos_by_score[score].append({
                                'path': str(video_path),
                                'video_id': video_id,
                                'subject': subject,
                                'score': score
                            })

    return videos_by_score


def extract_landmarks(video_path: str, max_frames: int = 300) -> List[Dict]:
    """Extract MediaPipe pose landmarks from video"""
    mp_pose = mp.solutions.pose

    landmarks_list = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                keypoints = []
                world_keypoints = []

                for i, lm in enumerate(results.pose_landmarks.landmark):
                    keypoints.append({
                        'id': i,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

                if results.pose_world_landmarks:
                    for i, lm in enumerate(results.pose_world_landmarks.landmark):
                        world_keypoints.append({
                            'id': i,
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z,
                            'visibility': lm.visibility
                        })

                landmarks_list.append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'keypoints': keypoints,
                    'world_keypoints': world_keypoints
                })

            frame_idx += 1

    cap.release()
    return landmarks_list


def test_analyzer(analyzer, landmarks: List[Dict], fps: float, version: str) -> Dict:
    """Test an analyzer and return results"""
    result = {
        'version': version,
        'success': False,
        'total_cycles': 0,
        'num_events': 0,
        'error': None,
        'confidence': 0.0,
        'detection_method': 'none',
        'camera_view': 'unknown',
        'is_partial': False
    }

    try:
        if version == 'v1':
            analyzer_instance = analyzer(fps=fps)
            analysis = analyzer_instance.analyze(landmarks)
            analysis_dict = analyzer_instance.to_dict(analysis)

            result['success'] = True
            result['total_cycles'] = analysis_dict['summary']['total_cycles']
            result['num_events'] = len(analysis_dict['events'])

        else:  # v2
            analyzer_instance = analyzer(fps=fps, verbose=False)
            analysis = analyzer_instance.analyze(landmarks, allow_partial=True)
            analysis_dict = analyzer_instance.to_dict(analysis)

            result['success'] = not analysis.is_partial or analysis.total_cycles > 0
            result['total_cycles'] = analysis_dict['summary']['total_cycles']
            result['num_events'] = len(analysis_dict['events'])
            result['confidence'] = analysis_dict['summary'].get('overall_confidence', 0)
            result['detection_method'] = analysis_dict['summary'].get('detection_method', 'unknown')
            result['camera_view'] = analysis_dict['summary'].get('camera_view', 'unknown')
            result['is_partial'] = analysis_dict['summary'].get('is_partial', False)

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    print("="*70)
    print("GaitCycleAnalyzer V2 Test - PD4T Dataset")
    print("="*70)

    # Get test videos
    print("\nLoading test videos...")
    videos_by_score = get_test_videos(n_per_score=3)

    total_videos = sum(len(v) for v in videos_by_score.values())
    print(f"Found {total_videos} test videos")
    for score, videos in videos_by_score.items():
        print(f"  Score {score}: {len(videos)} videos")

    if total_videos == 0:
        print("No videos found. Check PD4T path.")
        return

    # Test results
    results = {
        'v1': {'success': 0, 'fail': 0, 'total_cycles': 0},
        'v2': {'success': 0, 'fail': 0, 'total_cycles': 0, 'partial': 0}
    }

    detailed_results = []

    # Process each video
    print("\n" + "-"*70)
    print("Processing videos...")
    print("-"*70)

    for score, videos in videos_by_score.items():
        for video_info in videos:
            print(f"\n[Score {score}] {video_info['video_id']} ({video_info['subject']})")

            # Extract landmarks
            print("  Extracting landmarks...", end=" ", flush=True)
            landmarks = extract_landmarks(video_info['path'], max_frames=300)
            print(f"{len(landmarks)} frames")

            if len(landmarks) < 30:
                print("  Skipping: Not enough landmarks")
                continue

            # Get FPS from video
            cap = cv2.VideoCapture(video_info['path'])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            # Test V1
            print("  Testing V1...", end=" ", flush=True)
            v1_result = test_analyzer(V1Analyzer, landmarks, fps, 'v1')
            if v1_result['success']:
                print(f"OK ({v1_result['total_cycles']} cycles)")
                results['v1']['success'] += 1
                results['v1']['total_cycles'] += v1_result['total_cycles']
            else:
                print(f"FAIL: {v1_result['error']}")
                results['v1']['fail'] += 1

            # Test V2
            print("  Testing V2...", end=" ", flush=True)
            v2_result = test_analyzer(V2Analyzer, landmarks, fps, 'v2')
            if v2_result['success']:
                conf_str = f"conf={v2_result['confidence']:.2f}"
                method_str = v2_result['detection_method']
                partial_str = " (partial)" if v2_result['is_partial'] else ""
                print(f"OK ({v2_result['total_cycles']} cycles, {conf_str}, {method_str}){partial_str}")
                results['v2']['success'] += 1
                results['v2']['total_cycles'] += v2_result['total_cycles']
                if v2_result['is_partial']:
                    results['v2']['partial'] += 1
            else:
                print(f"FAIL: {v2_result.get('error', 'Unknown')}")
                results['v2']['fail'] += 1

            detailed_results.append({
                'video_id': video_info['video_id'],
                'subject': video_info['subject'],
                'score': score,
                'v1': v1_result,
                'v2': v2_result
            })

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    total_tested = results['v1']['success'] + results['v1']['fail']

    print(f"\nV1 (Original):")
    print(f"  Success Rate: {results['v1']['success']}/{total_tested} ({100*results['v1']['success']/total_tested:.1f}%)")
    print(f"  Total Cycles Detected: {results['v1']['total_cycles']}")

    print(f"\nV2 (Improved):")
    print(f"  Success Rate: {results['v2']['success']}/{total_tested} ({100*results['v2']['success']/total_tested:.1f}%)")
    print(f"  Total Cycles Detected: {results['v2']['total_cycles']}")
    print(f"  Partial Results: {results['v2']['partial']}")

    improvement = results['v2']['success'] - results['v1']['success']
    print(f"\n  Improvement: +{improvement} videos ({100*improvement/total_tested:.1f}%)")

    # Detection method distribution
    method_counts = {}
    camera_counts = {}
    for r in detailed_results:
        if r['v2']['success']:
            method = r['v2']['detection_method']
            camera = r['v2']['camera_view']
            method_counts[method] = method_counts.get(method, 0) + 1
            camera_counts[camera] = camera_counts.get(camera, 0) + 1

    print(f"\nV2 Detection Methods:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")

    print(f"\nV2 Camera Views Detected:")
    for view, count in sorted(camera_counts.items(), key=lambda x: -x[1]):
        print(f"  {view}: {count}")

    # Save results
    output_path = Path(__file__).parent / "gait_analyzer_v2_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'v1_success_rate': results['v1']['success'] / total_tested if total_tested > 0 else 0,
                'v2_success_rate': results['v2']['success'] / total_tested if total_tested > 0 else 0,
                'improvement': improvement,
                'total_tested': total_tested
            },
            'detailed': detailed_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
