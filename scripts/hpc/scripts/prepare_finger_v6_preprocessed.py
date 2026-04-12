"""
Finger Tapping v6 - 3D Sequence with Advanced Preprocessing
v5 + 전처리 (스무딩, 칼만필터, 보간법, 정규화)

전처리 파이프라인:
1. Missing value interpolation (결측치 보간)
2. Kalman filter (노이즈 제거)
3. Savitzky-Golay smoothing (스무딩)
4. Z-score normalization per feature (정규화)
5. Temporal alignment (시간 정렬)
"""
import os
import pickle
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

class KalmanFilter1D:
    """1D 칼만 필터 for landmark smoothing"""
    def __init__(self, process_variance=1e-4, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def filter(self, measurements):
        n = len(measurements)
        if n == 0:
            return measurements

        # 초기화
        x_est = measurements[0]  # 초기 추정
        p_est = 1.0  # 초기 오차 공분산

        filtered = np.zeros(n)

        for i in range(n):
            # 예측
            x_pred = x_est
            p_pred = p_est + self.process_variance

            # 업데이트
            k = p_pred / (p_pred + self.measurement_variance)  # 칼만 이득
            x_est = x_pred + k * (measurements[i] - x_pred)
            p_est = (1 - k) * p_pred

            filtered[i] = x_est

        return filtered


def interpolate_missing(signal_data, threshold=0.01):
    """결측치 보간 (0이나 NaN 값을 보간)"""
    signal_data = np.array(signal_data, dtype=float)

    # 유효한 값 찾기
    valid_mask = ~np.isnan(signal_data) & (np.abs(signal_data) > threshold)

    if np.sum(valid_mask) < 2:
        # 유효한 값이 2개 미만이면 원본 반환
        return signal_data

    valid_indices = np.where(valid_mask)[0]
    valid_values = signal_data[valid_mask]

    # 보간 함수 생성
    interp_func = interp1d(
        valid_indices, valid_values,
        kind='linear',
        bounds_error=False,
        fill_value=(valid_values[0], valid_values[-1])
    )

    # 전체 인덱스에 대해 보간
    all_indices = np.arange(len(signal_data))
    interpolated = interp_func(all_indices)

    return interpolated


def savgol_smooth(signal_data, window_length=11, polyorder=3):
    """Savitzky-Golay 필터로 스무딩"""
    if len(signal_data) < window_length:
        window_length = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1
        if window_length < 3:
            return signal_data

    return signal.savgol_filter(signal_data, window_length, polyorder)


def preprocess_sequence(X, apply_kalman=True, apply_savgol=True,
                       apply_interpolation=True, apply_gaussian=False):
    """
    단일 시퀀스 전처리

    Args:
        X: (n_frames, n_features) 시퀀스
        apply_kalman: 칼만 필터 적용
        apply_savgol: Savitzky-Golay 스무딩 적용
        apply_interpolation: 결측치 보간 적용
        apply_gaussian: 가우시안 스무딩 적용 (savgol 대신)
    """
    n_frames, n_features = X.shape
    X_processed = np.copy(X)

    kf = KalmanFilter1D(process_variance=1e-4, measurement_variance=0.05)

    for f in range(n_features):
        signal_data = X[:, f]

        # 1. 결측치 보간
        if apply_interpolation:
            # 0 값을 NaN으로 처리 (landmark 좌표에서 0은 보통 결측)
            if f < 63:  # landmark 좌표
                signal_data = np.where(signal_data == 0, np.nan, signal_data)
            signal_data = interpolate_missing(signal_data)

        # 2. 칼만 필터
        if apply_kalman:
            signal_data = kf.filter(signal_data)

        # 3. 스무딩
        if apply_savgol and len(signal_data) >= 11:
            signal_data = savgol_smooth(signal_data)
        elif apply_gaussian:
            signal_data = gaussian_filter1d(signal_data, sigma=1.5)

        X_processed[:, f] = signal_data

    return X_processed


def normalize_sequence(X, method='zscore'):
    """
    시퀀스 정규화

    Args:
        X: (n_frames, n_features)
        method: 'zscore', 'minmax', 'robust'
    """
    X_norm = np.copy(X)

    for f in range(X.shape[1]):
        feature = X[:, f]

        if method == 'zscore':
            mean = np.mean(feature)
            std = np.std(feature)
            if std > 1e-6:
                X_norm[:, f] = (feature - mean) / std
            else:
                X_norm[:, f] = feature - mean

        elif method == 'minmax':
            min_val = np.min(feature)
            max_val = np.max(feature)
            if max_val - min_val > 1e-6:
                X_norm[:, f] = (feature - min_val) / (max_val - min_val)
            else:
                X_norm[:, f] = 0.5

        elif method == 'robust':
            median = np.median(feature)
            q1, q3 = np.percentile(feature, [25, 75])
            iqr = q3 - q1
            if iqr > 1e-6:
                X_norm[:, f] = (feature - median) / iqr
            else:
                X_norm[:, f] = feature - median

    return X_norm


def temporal_alignment(X, target_length=150):
    """
    시간축 정렬 (리샘플링)

    Args:
        X: (n_frames, n_features)
        target_length: 목표 프레임 수
    """
    n_frames, n_features = X.shape

    if n_frames == target_length:
        return X

    # 보간을 통한 리샘플링
    original_indices = np.linspace(0, 1, n_frames)
    target_indices = np.linspace(0, 1, target_length)

    X_resampled = np.zeros((target_length, n_features))

    for f in range(n_features):
        interp_func = interp1d(original_indices, X[:, f], kind='linear')
        X_resampled[:, f] = interp_func(target_indices)

    return X_resampled


def preprocess_dataset(X, config=None):
    """
    전체 데이터셋 전처리

    Args:
        X: (n_samples, n_frames, n_features)
        config: 전처리 설정
    """
    if config is None:
        config = {
            'apply_interpolation': True,
            'apply_kalman': True,
            'apply_savgol': True,
            'apply_gaussian': False,
            'normalize_method': 'zscore',
            'target_length': 150
        }

    n_samples = X.shape[0]
    X_processed = []

    for i in tqdm(range(n_samples), desc="Preprocessing"):
        seq = X[i]

        # 1. 기본 전처리 (보간, 칼만, 스무딩)
        seq = preprocess_sequence(
            seq,
            apply_kalman=config['apply_kalman'],
            apply_savgol=config['apply_savgol'],
            apply_interpolation=config['apply_interpolation'],
            apply_gaussian=config['apply_gaussian']
        )

        # 2. 시간축 정렬
        if config['target_length']:
            seq = temporal_alignment(seq, config['target_length'])

        # 3. 정규화
        if config['normalize_method']:
            seq = normalize_sequence(seq, config['normalize_method'])

        X_processed.append(seq)

    return np.array(X_processed)


def main():
    print("=" * 70)
    print("Finger Tapping v6 - 3D Sequence with Advanced Preprocessing")
    print("=" * 70)

    # Load v5 data (v3 + breakpoint features)
    data_dir = './data'

    print("\nLoading v5 3D data...")
    with open(f'{data_dir}/finger_train_v5_3d.pkl', 'rb') as f:
        train_v5 = pickle.load(f)
    with open(f'{data_dir}/finger_valid_v5_3d.pkl', 'rb') as f:
        valid_v5 = pickle.load(f)
    with open(f'{data_dir}/finger_test_v5_3d.pkl', 'rb') as f:
        test_v5 = pickle.load(f)

    X_train = np.array(train_v5['X'])
    X_valid = np.array(valid_v5['X'])
    X_test = np.array(test_v5['X'])

    print(f"v5 Train shape: {X_train.shape}")
    print(f"v5 Valid shape: {X_valid.shape}")
    print(f"v5 Test shape: {X_test.shape}")

    # Preprocessing configurations
    configs = {
        'v6a_kalman_savgol': {
            'apply_interpolation': True,
            'apply_kalman': True,
            'apply_savgol': True,
            'apply_gaussian': False,
            'normalize_method': 'zscore',
            'target_length': 150
        },
        'v6b_kalman_only': {
            'apply_interpolation': True,
            'apply_kalman': True,
            'apply_savgol': False,
            'apply_gaussian': False,
            'normalize_method': 'zscore',
            'target_length': 150
        },
        'v6c_gaussian': {
            'apply_interpolation': True,
            'apply_kalman': False,
            'apply_savgol': False,
            'apply_gaussian': True,
            'normalize_method': 'zscore',
            'target_length': 150
        }
    }

    # 기본 설정으로 v6 생성 (kalman + savgol)
    config = configs['v6a_kalman_savgol']

    print(f"\nPreprocessing configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nApplying preprocessing...")
    print("\nProcessing train set:")
    X_train_v6 = preprocess_dataset(X_train, config)

    print("\nProcessing valid set:")
    X_valid_v6 = preprocess_dataset(X_valid, config)

    print("\nProcessing test set:")
    X_test_v6 = preprocess_dataset(X_test, config)

    print(f"\nv6 Train shape: {X_train_v6.shape}")
    print(f"v6 Valid shape: {X_valid_v6.shape}")
    print(f"v6 Test shape: {X_test_v6.shape}")

    # Handle NaN/inf
    X_train_v6 = np.nan_to_num(X_train_v6, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid_v6 = np.nan_to_num(X_valid_v6, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_v6 = np.nan_to_num(X_test_v6, nan=0.0, posinf=0.0, neginf=0.0)

    # Save v6 data
    train_v6 = {
        'X': X_train_v6,
        'y': train_v5['y'],
        'ids': train_v5['ids'],
        'feature_names': train_v5['feature_names'],
        'preprocessing_config': config
    }

    valid_v6 = {
        'X': X_valid_v6,
        'y': valid_v5['y'],
        'ids': valid_v5['ids'],
        'feature_names': valid_v5['feature_names'],
        'preprocessing_config': config
    }

    test_v6 = {
        'X': X_test_v6,
        'y': test_v5['y'],
        'ids': test_v5['ids'],
        'feature_names': test_v5['feature_names'],
        'preprocessing_config': config
    }

    with open(f'{data_dir}/finger_train_v6_preprocessed.pkl', 'wb') as f:
        pickle.dump(train_v6, f)
    with open(f'{data_dir}/finger_valid_v6_preprocessed.pkl', 'wb') as f:
        pickle.dump(valid_v6, f)
    with open(f'{data_dir}/finger_test_v6_preprocessed.pkl', 'wb') as f:
        pickle.dump(test_v6, f)

    print(f"\nSaved v6 preprocessed data:")
    print(f"  - {data_dir}/finger_train_v6_preprocessed.pkl")
    print(f"  - {data_dir}/finger_valid_v6_preprocessed.pkl")
    print(f"  - {data_dir}/finger_test_v6_preprocessed.pkl")

    # Statistics comparison
    print("\n" + "=" * 70)
    print("Preprocessing Statistics Comparison")
    print("=" * 70)

    print(f"\n{'Statistic':<20} {'Before (v5)':<20} {'After (v6)':<20}")
    print("-" * 60)

    # Sample statistics from first sample, first feature
    for i, fname in enumerate(['Mean', 'Std', 'Min', 'Max']):
        v5_val = [np.mean, np.std, np.min, np.max][i](X_train[0, :, 0])
        v6_val = [np.mean, np.std, np.min, np.max][i](X_train_v6[0, :, 0])
        print(f"{fname:<20} {v5_val:<20.4f} {v6_val:<20.4f}")

    print("\n" + "=" * 70)
    print("v6 Preprocessing Complete!")
    print("=" * 70)
    print("\nPreprocessing pipeline:")
    print("  1. Missing value interpolation (linear)")
    print("  2. Kalman filtering (noise reduction)")
    print("  3. Savitzky-Golay smoothing (window=11, order=3)")
    print("  4. Z-score normalization (per feature)")


if __name__ == "__main__":
    main()
