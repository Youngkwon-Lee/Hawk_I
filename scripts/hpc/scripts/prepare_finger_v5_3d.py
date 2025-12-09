"""
Finger Tapping v5 - 3D Sequence Features with Clinical Breakpoint Analysis
v3 기반 (568, 150, 98) + v4 breakpoint features를 3D로 확장

v3: 63 landmarks + 35 kinematic features (프레임별)
v5: v3 + 새로운 temporal breakpoint features (프레임별로 계산)
"""
import os
import pickle
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from tqdm import tqdm

def compute_frame_breakpoint_features(distances, frame_idx, window=30):
    """프레임별 breakpoint 및 temporal features 계산 (15개)"""
    n_frames = len(distances)

    # 현재 프레임 주변 window 내의 데이터 사용
    start = max(0, frame_idx - window // 2)
    end = min(n_frames, frame_idx + window // 2)
    local_dist = distances[start:end]

    if len(local_dist) < 5:
        return np.zeros(15)  # 15개의 새 features

    features = []

    # 1. Local amplitude statistics (3개)
    features.append(np.mean(local_dist))  # local_amp_mean
    features.append(np.std(local_dist))   # local_amp_std
    features.append(np.max(local_dist) - np.min(local_dist))  # local_range

    # 2. Temporal position features (1개)
    rel_pos = frame_idx / n_frames  # 상대적 위치 (0~1)
    features.append(rel_pos)  # relative_position

    # 3. Trend features (2개)
    if frame_idx < n_frames // 2:
        features.append(1.0)  # is_first_half
        features.append(0.0)  # is_second_half
    else:
        features.append(0.0)
        features.append(1.0)

    # 4. Local velocity (2개)
    if len(local_dist) > 1:
        local_vel = np.diff(local_dist)
        features.append(np.mean(local_vel))  # local_velocity_mean
        features.append(np.std(local_vel))   # local_velocity_std
    else:
        features.append(0.0)
        features.append(0.0)

    # 5. Local acceleration (2개)
    if len(local_dist) > 2:
        local_acc = np.diff(local_dist, n=2)
        features.append(np.mean(local_acc))  # local_acc_mean
        features.append(np.std(local_acc))   # local_acc_std
    else:
        features.append(0.0)
        features.append(0.0)

    # 6. Cumulative features (1개)
    cumsum = np.cumsum(distances[:frame_idx+1]) if frame_idx > 0 else np.array([distances[0]])
    features.append(cumsum[-1] / (frame_idx + 1))  # cumulative_mean

    # 7. Deviation from expected (1개)
    expected = distances[0] * (1 - 0.3 * rel_pos)  # 30% decay expected
    actual = distances[frame_idx]
    features.append(actual - expected)  # deviation_from_expected

    # 8. Local peak/trough indicator (2개)
    if frame_idx > 0 and frame_idx < n_frames - 1:
        is_peak = distances[frame_idx] > distances[frame_idx-1] and distances[frame_idx] > distances[frame_idx+1]
        is_trough = distances[frame_idx] < distances[frame_idx-1] and distances[frame_idx] < distances[frame_idx+1]
        features.append(1.0 if is_peak else 0.0)   # is_local_peak
        features.append(1.0 if is_trough else 0.0) # is_local_trough
    else:
        features.append(0.0)
        features.append(0.0)

    # 9. Local smoothness (1개) - 총 15개
    if len(local_dist) > 3:
        smoothness = np.mean(np.abs(np.diff(local_dist, n=2)))
        features.append(smoothness)  # local_smoothness
    else:
        features.append(0.0)

    return np.array(features)


def compute_global_breakpoint_features(distances):
    """전체 시퀀스에서 계산되는 global breakpoint features (모든 프레임에 동일하게 적용)"""
    n_frames = len(distances)

    if n_frames < 10:
        return np.zeros(10)

    features = []

    # 1. Breakpoint detection (선형 회귀의 변곡점)
    best_breakpoint = n_frames // 2
    best_mse = float('inf')

    for bp in range(n_frames // 4, 3 * n_frames // 4):
        # 두 구간으로 나누어 선형 회귀
        x1 = np.arange(bp)
        x2 = np.arange(bp, n_frames)

        if len(x1) < 3 or len(x2) < 3:
            continue

        y1 = distances[:bp]
        y2 = distances[bp:]

        # 각 구간 선형 회귀
        slope1, intercept1 = np.polyfit(x1, y1, 1)
        slope2, intercept2 = np.polyfit(x2 - bp, y2, 1)

        # MSE 계산
        pred1 = slope1 * x1 + intercept1
        pred2 = slope2 * (x2 - bp) + intercept2
        mse = np.mean((y1 - pred1)**2) + np.mean((y2 - pred2)**2)

        if mse < best_mse:
            best_mse = mse
            best_breakpoint = bp
            best_slope1 = slope1
            best_slope2 = slope2

    features.append(best_breakpoint / n_frames)  # breakpoint_norm
    features.append(best_slope1 if 'best_slope1' in dir() else 0)  # slope1
    features.append(best_slope2 if 'best_slope2' in dir() else 0)  # slope2

    # 2. Exponential decay fitting
    try:
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        x = np.arange(n_frames)
        popt, _ = curve_fit(exp_decay, x, distances,
                           p0=[distances[0], 0.01, distances[-1]],
                           maxfev=1000)

        pred = exp_decay(x, *popt)
        r2 = 1 - np.sum((distances - pred)**2) / np.sum((distances - np.mean(distances))**2)

        features.append(popt[1])  # exp_decay_rate
        features.append(max(0, r2))  # exp_r2
    except:
        features.append(0.0)
        features.append(0.0)

    # 3. First half vs Second half comparison
    first_half = distances[:n_frames//2]
    second_half = distances[n_frames//2:]

    features.append(np.mean(first_half))  # first_half_mean
    features.append(np.mean(second_half))  # second_half_mean
    features.append(np.mean(first_half) / (np.mean(second_half) + 1e-6))  # half_ratio

    # 4. Rhythm regularity (autocorrelation based)
    if len(distances) > 20:
        autocorr = np.correlate(distances - np.mean(distances),
                               distances - np.mean(distances), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-6)

        # Find dominant frequency
        peaks, _ = signal.find_peaks(autocorr, height=0.3)
        if len(peaks) > 0:
            features.append(peaks[0] / n_frames)  # rhythm_period_norm
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # 5. Sequence trend (overall direction)
    slope, _ = np.polyfit(np.arange(n_frames), distances, 1)
    features.append(slope)  # sequence_trend

    return np.array(features)


def extract_v5_features(X_v3, distances_list=None):
    """
    v3 features (568, 150, 98)에 새로운 temporal breakpoint features 추가
    Output: (568, 150, 98 + 15 + 10) = (568, 150, 123)
    """
    n_samples, n_frames, n_v3_features = X_v3.shape

    # 새로운 features: 15 frame-level + 10 global (broadcast)
    n_new_features = 25
    X_v5 = np.zeros((n_samples, n_frames, n_v3_features + n_new_features))

    for i in tqdm(range(n_samples), desc="Extracting v5 features"):
        # v3 features 복사
        X_v5[i, :, :n_v3_features] = X_v3[i]

        # thumb-index distance 계산 (v3의 kinematic features에서 추출)
        # v3 kinematic features 중 거리 관련 feature 사용
        # 또는 raw landmark에서 계산

        # v3에서 거리 시퀀스 추출 (landmark 좌표에서)
        # thumb_tip = landmark 4, index_tip = landmark 8
        # 각 landmark는 3D (x, y, z)
        # v3: 63 landmark coords (21 * 3) + 35 kinematic

        # thumb_tip: indices 12, 13, 14 (4*3, 4*3+1, 4*3+2)
        # index_tip: indices 24, 25, 26 (8*3, 8*3+1, 8*3+2)

        thumb_xyz = X_v3[i, :, 12:15]  # (150, 3)
        index_xyz = X_v3[i, :, 24:27]  # (150, 3)

        # Euclidean distance per frame
        distances = np.sqrt(np.sum((thumb_xyz - index_xyz)**2, axis=1))

        # Handle NaN/zeros
        distances = np.nan_to_num(distances, nan=0.0)
        if np.sum(distances) == 0:
            distances = np.ones(n_frames) * 0.1

        # Global breakpoint features (모든 프레임에 동일)
        global_features = compute_global_breakpoint_features(distances)

        # Frame-level features
        for f in range(n_frames):
            frame_features = compute_frame_breakpoint_features(distances, f)

            # 새 features 추가
            X_v5[i, f, n_v3_features:n_v3_features+15] = frame_features
            X_v5[i, f, n_v3_features+15:] = global_features

    return X_v5


def main():
    print("=" * 70)
    print("Finger Tapping v5 - 3D Sequence with Breakpoint Features")
    print("=" * 70)

    # Load v3 data
    data_dir = './data'

    print("\nLoading v3 data...")
    with open(f'{data_dir}/finger_train_v3.pkl', 'rb') as f:
        train_v3 = pickle.load(f)
    with open(f'{data_dir}/finger_valid_v3.pkl', 'rb') as f:
        valid_v3 = pickle.load(f)
    with open(f'{data_dir}/finger_test_v3.pkl', 'rb') as f:
        test_v3 = pickle.load(f)

    print(f"v3 Train shape: {np.array(train_v3['X']).shape}")
    print(f"v3 Valid shape: {np.array(valid_v3['X']).shape}")
    print(f"v3 Test shape: {np.array(test_v3['X']).shape}")

    # Convert to numpy
    X_train = np.array(train_v3['X'])
    X_valid = np.array(valid_v3['X'])
    X_test = np.array(test_v3['X'])

    # Extract v5 features
    print("\nExtracting v5 features (v3 + breakpoint)...")
    X_train_v5 = extract_v5_features(X_train)
    X_valid_v5 = extract_v5_features(X_valid)
    X_test_v5 = extract_v5_features(X_test)

    print(f"\nv5 Train shape: {X_train_v5.shape}")
    print(f"v5 Valid shape: {X_valid_v5.shape}")
    print(f"v5 Test shape: {X_test_v5.shape}")

    # Feature names
    v3_feature_names = [f'v3_feat_{i}' for i in range(98)]

    frame_feature_names = [
        'local_amp_mean', 'local_amp_std', 'local_range',
        'relative_position', 'is_first_half', 'is_second_half',
        'local_velocity_mean', 'local_velocity_std',
        'local_acc_mean', 'local_acc_std',
        'cumulative_mean', 'deviation_from_expected',
        'is_local_peak', 'is_local_trough', 'local_smoothness'  # 15개
    ]

    global_feature_names = [
        'breakpoint_norm', 'slope1', 'slope2',
        'exp_decay_rate', 'exp_r2',
        'first_half_mean', 'second_half_mean', 'half_ratio',
        'rhythm_period_norm', 'sequence_trend'
    ]

    all_feature_names = v3_feature_names + frame_feature_names + global_feature_names

    # Save v5 data
    train_v5 = {
        'X': X_train_v5,
        'y': train_v3['y'],
        'ids': train_v3['ids'],
        'feature_names': all_feature_names
    }

    valid_v5 = {
        'X': X_valid_v5,
        'y': valid_v3['y'],
        'ids': valid_v3['ids'],
        'feature_names': all_feature_names
    }

    test_v5 = {
        'X': X_test_v5,
        'y': test_v3['y'],
        'ids': test_v3['ids'],
        'feature_names': all_feature_names
    }

    os.makedirs(data_dir, exist_ok=True)

    with open(f'{data_dir}/finger_train_v5_3d.pkl', 'wb') as f:
        pickle.dump(train_v5, f)
    with open(f'{data_dir}/finger_valid_v5_3d.pkl', 'wb') as f:
        pickle.dump(valid_v5, f)
    with open(f'{data_dir}/finger_test_v5_3d.pkl', 'wb') as f:
        pickle.dump(test_v5, f)

    print(f"\nSaved v5 3D data:")
    print(f"  - {data_dir}/finger_train_v5_3d.pkl")
    print(f"  - {data_dir}/finger_valid_v5_3d.pkl")
    print(f"  - {data_dir}/finger_test_v5_3d.pkl")

    print(f"\nFeature breakdown:")
    print(f"  v3 features: 98 (63 landmarks + 35 kinematic)")
    print(f"  Frame-level breakpoint: 15")
    print(f"  Global breakpoint: 10")
    print(f"  Total: {X_train_v5.shape[2]}")

    print("\n" + "=" * 70)
    print("v5 3D Feature Extraction Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
