"""
Compare 3 camera-independent arm swing calculation methods
"""
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

PD4T_ROOT = 'C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait'

videos = {
    'Score 2 (Subject 27)': f'{PD4T_ROOT}/027/15-002970.mp4',
    'Score 3 (Subject 4)': f'{PD4T_ROOT}/004/13-007586.mp4'
}

def calculate_arm_swing_methods(video_path, max_frames=300):
    """Calculate arm swing using 3 different methods"""
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    # Store trajectories
    left_wrist_3d = []
    right_wrist_3d = []
    left_shoulder_3d = []
    right_shoulder_3d = []
    left_elbow_3d = []
    right_elbow_3d = []
    left_hip_3d = []
    right_hip_3d = []

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark

            # Get 3D positions (World Landmarks - hip-centered, meters)
            left_wrist_3d.append([lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                   lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                   lm[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
            right_wrist_3d.append([lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                    lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
            left_shoulder_3d.append([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
            right_shoulder_3d.append([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
            left_elbow_3d.append([lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                   lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                   lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
            right_elbow_3d.append([lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
            left_hip_3d.append([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                 lm[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                 lm[mp_pose.PoseLandmark.LEFT_HIP.value].z])
            right_hip_3d.append([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                  lm[mp_pose.PoseLandmark.RIGHT_HIP.value].z])

        frame_count += 1

    cap.release()
    pose.close()

    if len(left_wrist_3d) < 50:
        return None

    # Convert to numpy arrays
    left_wrist = np.array(left_wrist_3d)
    right_wrist = np.array(right_wrist_3d)
    left_shoulder = np.array(left_shoulder_3d)
    right_shoulder = np.array(right_shoulder_3d)
    left_elbow = np.array(left_elbow_3d)
    right_elbow = np.array(right_elbow_3d)
    left_hip = np.array(left_hip_3d)
    right_hip = np.array(right_hip_3d)

    results = {}

    # ==========================================
    # Method 1: Body-Centered Coordinate System
    # ==========================================
    # Transform to walking direction coordinate system
    # Walking direction = average hip movement direction
    hip_center = (left_hip + right_hip) / 2

    # Calculate walking direction from hip movement
    hip_velocity = np.diff(hip_center, axis=0)
    walking_dir = np.mean(hip_velocity, axis=0)
    walking_dir[1] = 0  # Ignore vertical component
    if np.linalg.norm(walking_dir) > 0.001:
        walking_dir = walking_dir / np.linalg.norm(walking_dir)
    else:
        walking_dir = np.array([0, 0, 1])  # Default forward

    # Perpendicular direction (arm swing direction)
    arm_swing_dir = np.array([-walking_dir[2], 0, walking_dir[0]])

    # Project wrist movement onto arm swing direction
    left_wrist_rel = left_wrist - hip_center
    right_wrist_rel = right_wrist - hip_center

    left_proj = np.dot(left_wrist_rel, arm_swing_dir)
    right_proj = np.dot(right_wrist_rel, arm_swing_dir)

    left_amp_m1 = np.ptp(left_proj)
    right_amp_m1 = np.ptp(right_proj)

    results['method1_body_centered'] = {
        'left': left_amp_m1,
        'right': right_amp_m1,
        'mean': (left_amp_m1 + right_amp_m1) / 2
    }

    # ==========================================
    # Method 2: Joint Angle Based (Shoulder angle)
    # ==========================================
    # Angle between upper arm (shoulder-elbow) and trunk (shoulder-hip)
    def angle_between_vectors(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    left_angles = []
    right_angles = []

    for i in range(len(left_shoulder)):
        # Left arm
        trunk_vec = left_hip[i] - left_shoulder[i]
        arm_vec = left_elbow[i] - left_shoulder[i]
        left_angles.append(angle_between_vectors(trunk_vec, arm_vec))

        # Right arm
        trunk_vec = right_hip[i] - right_shoulder[i]
        arm_vec = right_elbow[i] - right_shoulder[i]
        right_angles.append(angle_between_vectors(trunk_vec, arm_vec))

    left_angle_range = np.ptp(left_angles)
    right_angle_range = np.ptp(right_angles)

    results['method2_joint_angle'] = {
        'left': left_angle_range,
        'right': right_angle_range,
        'mean': (left_angle_range + right_angle_range) / 2
    }

    # ==========================================
    # Method 3: 3D Magnitude (Total displacement)
    # ==========================================
    # Wrist position relative to shoulder (removes body movement)
    left_wrist_rel_shoulder = left_wrist - left_shoulder
    right_wrist_rel_shoulder = right_wrist - right_shoulder

    # Calculate 3D distance from mean position
    left_mean = np.mean(left_wrist_rel_shoulder, axis=0)
    right_mean = np.mean(right_wrist_rel_shoulder, axis=0)

    left_distances = np.linalg.norm(left_wrist_rel_shoulder - left_mean, axis=1)
    right_distances = np.linalg.norm(right_wrist_rel_shoulder - right_mean, axis=1)

    # Peak-to-peak of distances (max swing amplitude)
    left_amp_m3 = np.max(left_distances) * 2  # Approximate total swing
    right_amp_m3 = np.max(right_distances) * 2

    results['method3_3d_magnitude'] = {
        'left': left_amp_m3,
        'right': right_amp_m3,
        'mean': (left_amp_m3 + right_amp_m3) / 2
    }

    # ==========================================
    # Current Method (Z-axis only) for comparison
    # ==========================================
    left_z_range = np.ptp(left_wrist[:, 2] - hip_center[:, 2])
    right_z_range = np.ptp(right_wrist[:, 2] - hip_center[:, 2])

    results['current_z_axis'] = {
        'left': left_z_range,
        'right': right_z_range,
        'mean': (left_z_range + right_z_range) / 2
    }

    return results

def main():
    print('='*70)
    print('Comparing Arm Swing Calculation Methods')
    print('='*70)
    print('\nGoal: Score 3 should have LESS arm swing than Score 2')
    print('      (Lower value = worse PD symptoms)')

    all_results = {}

    for label, path in videos.items():
        print(f'\nProcessing {label}...')
        results = calculate_arm_swing_methods(path)

        if results:
            all_results[label] = results
        else:
            print(f'  ERROR: Not enough frames')

    # Compare results
    print('\n' + '='*70)
    print('RESULTS COMPARISON')
    print('='*70)

    methods = ['current_z_axis', 'method1_body_centered', 'method2_joint_angle', 'method3_3d_magnitude']
    method_names = ['Current (Z-axis)', 'Body-Centered', 'Joint Angle (deg)', '3D Magnitude']

    for method, name in zip(methods, method_names):
        print(f'\n{name}:')
        print('-'*50)

        s2_val = all_results.get('Score 2 (Subject 27)', {}).get(method, {}).get('mean', 0)
        s3_val = all_results.get('Score 3 (Subject 4)', {}).get(method, {}).get('mean', 0)

        correct = 'YES' if s3_val < s2_val else 'NO'
        diff_pct = ((s2_val - s3_val) / s2_val * 100) if s2_val > 0 else 0

        print(f'  Score 2: {s2_val:.4f}')
        print(f'  Score 3: {s3_val:.4f}')
        print(f'  Score 3 < Score 2? {correct}')
        print(f'  Difference: {diff_pct:.1f}%')

if __name__ == '__main__':
    main()
