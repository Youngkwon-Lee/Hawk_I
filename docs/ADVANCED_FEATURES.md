# Advanced Feature Engineering for Parkinson's Assessment

## Current Features (Mamba + Enhanced)
- Original skeleton features (10)
- Velocity (1st derivative)
- Acceleration (2nd derivative)
- Moving statistics (mean, std, min, max)

**Total: 70 features → Pearson 0.609**

---

## Proposed Advanced Features

### 1. Biomechanical Features (High Priority)

#### A. Finger Tapping Specific
```python
# Amplitude Features
- peak_to_peak_amplitude: 손가락 벌림 최대 거리
- amplitude_decline_rate: 시간에 따른 진폭 감소 (fatigue)
- amplitude_coefficient_variation: CV of amplitude
- normalized_amplitude: 손 크기 대비 정규화

# Frequency Features
- dominant_frequency: FFT peak frequency
- frequency_variability: 주파수 변동성
- rhythm_consistency: 리듬 일관성 (autocorrelation)
- inter_tap_interval_cv: Tap 간격 변동계수

# Movement Quality
- smoothness: Jerk 기반 부드러움 (SPARC metric)
- movement_symmetry: 좌우 대칭성
- trajectory_curvature: 궤적의 곡률
- hesitation_count: 멈칫거림 횟수 (velocity < threshold)

# Clinical Markers
- bradykinesia_score: 느린 움직임 정도
- freezing_episodes: Freezing 에피소드 수
- fatigue_index: 초반 vs 후반 성능 비율
- decrement_score: UPDRS 기준 감소 점수
```

**예상 효과:** +5-10% Pearson

---

#### B. Joint Angle Features
```python
# MediaPipe로 3D 좌표 추출 후 각도 계산
- thumb_index_angle: 엄지-검지 각도
- wrist_flexion_angle: 손목 굴곡 각도
- angular_velocity: 각속도
- angular_acceleration: 각가속도

# Joint-specific kinematics
- MCP_joint_angle: Metacarpophalangeal joint
- PIP_joint_angle: Proximal interphalangeal joint
- DIP_joint_angle: Distal interphalangeal joint
```

**예상 효과:** +3-5% Pearson

---

### 2. Temporal Pattern Features (Medium Priority)

#### A. Rhythm & Periodicity
```python
# Autocorrelation features
- autocorr_peak_lag: 주기성 피크 시간
- autocorr_peak_value: 자기상관 최댓값
- periodicity_strength: 주기성 강도

# Spectral features
- spectral_entropy: 스펙트럼 엔트로피
- spectral_centroid: 스펙트럼 중심
- spectral_rolloff: 스펙트럼 롤오프
- mel_frequency_cepstral_coefficients: MFCC (음성 분석에서 차용)

# Pattern recognition
- repetition_consistency: 반복 일관성
- pattern_regularity: 패턴 규칙성
```

**예상 효과:** +2-4% Pearson

---

#### B. Wavelet Features
```python
# Multi-scale analysis
- wavelet_energy: 웨이블릿 에너지
- wavelet_entropy: 웨이블릿 엔트로피
- scale_dependent_features: 스케일별 특징
```

**예상 효과:** +1-3% Pearson

---

### 3. Clinical Domain Features (High Priority)

#### A. UPDRS-aligned Features
```python
# MDS-UPDRS 3.4 Finger Tapping criteria
- amplitude_score: 진폭 점수 (0-4)
- speed_score: 속도 점수
- rhythm_score: 리듬 점수
- halts_score: 멈춤 점수
- decreasing_amplitude_score: 진폭 감소 점수

# Composite scores
- motor_fluctuation_index: 운동 변동 지수
- on_off_ratio: On/Off 상태 비율 (약물 효과)
```

**예상 효과:** +5-8% Pearson

---

#### B. Fatigue & Deterioration
```python
# Temporal deterioration
- early_vs_late_ratio: 초반/후반 성능 비
- linear_decay_slope: 선형 감소 기울기
- exponential_decay_rate: 지수 감소율
- fatigue_onset_time: 피로 시작 시점

# Movement quality decay
- smoothness_decay: 부드러움 감소
- accuracy_decay: 정확도 감소
```

**예상 효과:** +3-5% Pearson

---

### 4. Multi-modal Features (Future)

#### A. RGB Video Features
```python
# Appearance-based
- hand_tremor_frequency: 손 떨림 주파수 (optical flow)
- skin_color_variation: 피부색 변화
- muscle_tension_proxy: 근육 긴장도 추정

# Motion-based
- optical_flow_magnitude: 광학 흐름 크기
- motion_energy: 움직임 에너지
```

**예상 효과:** +10-15% Pearson

---

#### B. Audio Features (if available)
```python
# Sound of tapping
- tap_sound_energy: Tap 소리 에너지
- tap_consistency: 소리 일관성
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Amplitude decline rate** - 파킨슨 핵심 지표
2. **Frequency variability** - FFT 기반, 구현 쉬움
3. **Smoothness (SPARC)** - Jerk 기반, 이미 있음
4. **Hesitation count** - Velocity threshold

**Expected Gain:** +5-7% Pearson → **0.64**

---

### Phase 2: Clinical Features (3-5 days)
1. **UPDRS-aligned scores** - 도메인 지식 활용
2. **Fatigue index** - 시간 구간 분석
3. **Joint angles** - MediaPipe 3D 활용

**Expected Gain:** +8-12% Pearson → **0.68**

---

### Phase 3: Advanced (1-2 weeks)
1. **VideoMamba** - RGB 비디오 직접 처리
2. **Wavelet features** - Multi-scale 분석
3. **Multi-task learning** - Finger + Gait 동시

**Expected Gain:** +15-20% Pearson → **0.72+**

---

## Code Snippets

### SPARC (Smoothness) Implementation
```python
import numpy as np
from scipy.signal import welch

def calculate_sparc(velocity):
    """
    Spectral Arc Length (SPARC) - Movement smoothness
    Lower is smoother
    """
    f, Pxx = welch(velocity, fs=30, nperseg=min(256, len(velocity)))

    # Normalize
    Pxx_norm = Pxx / np.sum(Pxx)

    # Arc length
    sparc = -np.sum(np.sqrt(np.diff(f)**2 + np.diff(Pxx_norm)**2))

    return sparc
```

### Amplitude Decline Rate
```python
def calculate_amplitude_decline(positions, window_size=10):
    """
    Calculate amplitude decline rate (fatigue indicator)
    """
    # Extract thumb-index distance per frame
    distances = np.linalg.norm(
        positions[:, thumb_idx] - positions[:, index_idx],
        axis=1
    )

    # Peak detection
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(distances)

    if len(peaks) < 2:
        return 0.0

    # Fit linear regression to peak amplitudes
    peak_amps = distances[peaks]
    decline_rate = np.polyfit(np.arange(len(peak_amps)), peak_amps, 1)[0]

    return decline_rate
```

### Hesitation Detection
```python
def detect_hesitations(velocity, threshold=0.1):
    """
    Detect hesitation episodes (freezing)
    """
    # Velocity magnitude
    vel_mag = np.linalg.norm(velocity, axis=1)

    # Below threshold
    hesitations = vel_mag < threshold

    # Count episodes
    from scipy.ndimage import label
    labeled, num_episodes = label(hesitations)

    return num_episodes, np.sum(hesitations) / len(hesitations)
```

---

## Expected Results Summary

| Phase | Features | Pearson | MAE | Exact |
|-------|----------|---------|-----|-------|
| Current | Enhanced (70) | 0.609 | 0.444 | 63.0% |
| Phase 1 | + Quick Wins | **0.64** | 0.43 | 64% |
| Phase 2 | + Clinical | **0.68** | 0.41 | 66% |
| Phase 3 | + VideoMamba | **0.72+** | 0.38 | 68% |

---

## References

1. **SPARC metric**: Balasubramanian et al. (2015) - "On the analysis of movement smoothness"
2. **Amplitude decline**: Bologna et al. (2016) - "Bradykinesia in Parkinson's disease"
3. **Finger tapping kinematics**: Espay et al. (2011) - "Differential response of speed, amplitude, and rhythm to dopaminergic medications in Parkinson's disease"
4. **MDS-UPDRS**: Goetz et al. (2008) - "Movement Disorder Society-UPDRS"
