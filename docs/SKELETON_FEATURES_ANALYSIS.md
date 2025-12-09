# Skeleton & Features 분석 문서

**작성일**: 2024-12-09
**목적**: MediaPipe 스켈레톤 추출 및 Feature 계산 방식 정리

---

## 1. MediaPipe Keypoints 정의

### 1.1 Hand Landmarks (21개) - Finger Tapping용

```
                    THUMB           INDEX          MIDDLE         RING           PINKY

                      4               8              12             16             20
                      │               │              │              │              │
                      3               7              11             15             19
                      │               │              │              │              │
                      2               6              10             14             18
                      │               │              │              │              │
                      1               5              9              13             17
                       \             /              /              /              /
                        \           /              /              /              /
                         \         /              /              /              /
                          ========0 (WRIST)========================
```

| ID | Keypoint | 설명 |
|----|----------|------|
| 0 | WRIST | 손목 |
| 1-4 | THUMB | 엄지 (CMC→MCP→IP→TIP) |
| 5-8 | INDEX | 검지 (MCP→PIP→DIP→TIP) |
| 9-12 | MIDDLE | 중지 |
| 13-16 | RING | 약지 |
| 17-20 | PINKY | 소지 |

**Finger Tapping 핵심 포인트**:
- `4` (THUMB_TIP) ↔ `8` (INDEX_TIP): 탭핑 거리 측정
- `5-8` (INDEX finger): 정규화 기준 길이
- `0` (WRIST): 기준점

### 1.2 Pose Landmarks (33개) - Gait용

```
                        0 (nose)
                       /|\
                    1-2 3-4 (eyes, ears)
                       │
                   11──┼──12  (shoulders)
                  /    │    \
                13    │     14  (elbows)
                /      │      \
               15     │       16  (wrists)
              /│\     │       /│\
           17-22     │      (hand detail)
                     │
                 23──┼──24  (hips)
                  \  │  /
                   \ │ /
                25  │  26  (knees)
                 \  │  /
                  \ │ /
                27  │  28  (ankles)
                 │  │  │
                29  │  30  (heels)
                 │     │
                31    32  (foot index/toes)
```

| ID | Keypoint | Gait 사용 |
|----|----------|-----------|
| 0 | NOSE | ✅ 머리 위치 |
| 11 | LEFT_SHOULDER | ✅ 상체 기울기 |
| 12 | RIGHT_SHOULDER | ✅ 상체 기울기 |
| 23 | LEFT_HIP | ✅ 골반 |
| 24 | RIGHT_HIP | ✅ 골반 |
| 25 | LEFT_KNEE | ✅ 무릎 각도 |
| 26 | RIGHT_KNEE | ✅ 무릎 각도 |
| 27 | LEFT_ANKLE | ✅ 보폭, 발목 |
| 28 | RIGHT_ANKLE | ✅ 보폭, 발목 |
| 31 | LEFT_FOOT_INDEX | ✅ 발끝 |
| 32 | RIGHT_FOOT_INDEX | ✅ 발끝 |
| 15 | LEFT_WRIST | ✅ 팔 스윙 |
| 16 | RIGHT_WRIST | ✅ 팔 스윙 |

---

## 2. 2D vs 3D 좌표 시스템

### 2.1 MediaPipe 출력 좌표

| 좌표 | 범위 | 설명 |
|------|------|------|
| `x` | 0.0 ~ 1.0 | 이미지 너비 정규화 |
| `y` | 0.0 ~ 1.0 | 이미지 높이 정규화 |
| `z` | 상대값 | 깊이 (기준점 대비) |
| `visibility` | 0.0 ~ 1.0 | 가시성/신뢰도 |

### 2.2 World Landmarks (Pose만 지원)

| 좌표 | 단위 | 설명 |
|------|------|------|
| `x` | meters | 실제 좌우 위치 |
| `y` | meters | 실제 상하 위치 |
| `z` | meters | 실제 전후 위치 |
| Origin | hip center | 골반 중심이 원점 |

### 2.3 Hand vs Pose 비교

| 기능 | MediaPipe Hands | MediaPipe Pose |
|------|-----------------|----------------|
| `landmarks` | ✅ x, y, z | ✅ x, y, z |
| `world_landmarks` | ❌ 미지원 | ✅ 실제 미터 단위 |
| 기준점 | 손목 (상대) | 골반 중심 (절대) |

**Hand에 world_landmarks가 없는 이유**:
1. 설계 목적: 제스처 인식 (상대 위치로 충분)
2. 손목이 기준점이라 손 전체 이동 시 의미 없음
3. Google 구현 결정

---

## 3. 현재 저장된 Skeleton 파일

### 3.1 파일 위치 및 포맷

| 위치 | 포맷 | z 좌표 | world_landmarks |
|------|------|--------|-----------------|
| `experiments/results/test_outputs/*.json` | 새 포맷 | ✅ | ✅ |
| `frontend/public/data/*.json` | 구 포맷 | ❌ | ❌ |

### 3.2 새 포맷 (mediapipe_processor.py 출력)
```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "keypoints": [
    {"id": 0, "x": 0.69, "y": 0.19, "z": -0.10, "visibility": 0.99}
  ],
  "world_keypoints": [
    {"id": 0, "x": 0.05, "y": -0.32, "z": 0.12, "visibility": 0.99}
  ]
}
```

### 3.3 구 포맷 (frontend용, 구버전)
```json
{
  "frame": 0,
  "keypoints": [
    {"id": 0, "x": 0.69, "y": 0.19, "score": 0.99}
  ]
}
```

---

## 4. Feature 측정 방식 분석

### 4.1 Finger Tapping Features (27개) - 모두 상대적 ✅

| Feature | 측정 방식 | 정규화 방법 |
|---------|----------|------------|
| `tapping_speed` | Hz | count / duration |
| `amplitude_mean/std` | dimensionless | **검지 길이로 정규화** |
| `amplitude_decrement` | % | 백분율 변화 |
| `first/second_half_amplitude` | dimensionless | 검지 길이 정규화 |
| `opening_velocity_mean` | units/s | 정규화 거리의 미분 |
| `closing_velocity_mean` | units/s | 정규화 거리의 미분 |
| `peak_velocity_mean` | units/s | 정규화 거리의 미분 |
| `velocity_decrement` | % | 백분율 |
| `rhythm_variability` | CV % | 변동계수 |
| `hesitation_count` | count | 횟수 |
| `halt_count` | count | 횟수 |
| `freeze_episodes` | count | 횟수 |
| `fatigue_rate` | %/tap | 평균 대비 % |
| `velocity_first/mid/last_third` | units/s | 정규화 |
| `amplitude_first/mid/last_third` | dimensionless | 정규화 |
| `velocity_slope` | %/tap | 평균 대비 % |
| `amplitude_slope` | %/tap | 평균 대비 % |
| `rhythm_slope` | CV/tap | CV 변화율 |
| `variability_first/second_half` | CV % | 변동계수 |
| `variability_change` | % | 백분율 |

#### 정규화 코드 (metrics_calculator.py)
```python
# 검지 손가락 길이 계산 (3D Euclidean)
seg1 = np.linalg.norm(index_pip - index_mcp)  # MCP→PIP
seg2 = np.linalg.norm(index_dip - index_pip)  # PIP→DIP
seg3 = np.linalg.norm(index_tip - index_dip)  # DIP→TIP
index_finger_length = seg1 + seg2 + seg3

# 정규화 (VisionMD 방법론)
norm_factor = np.max(finger_lengths)
distances_normalized = distances / norm_factor
```

#### 연구 레퍼런스

| 출처 | 정규화 방법 | 링크 |
|------|------------|------|
| **VisionMD (2025)** | Hand's length (손 전체 길이) | [Nature npj PD](https://www.nature.com/articles/s41531-025-00876-6) |
| **PMC10674854 (2023)** | Palm size (손바닥 크기) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10674854/) |
| **PDMotorDB** | Thumb length (엄지 길이) | 정면 촬영 시 과소추정 문제 |
| **MDS-UPDRS** | Clinical scoring criteria | Movement Disorder Society |

**우리 구현**: 검지 손가락 길이 (index finger length) = VisionMD 변형 버전
- VisionMD: 전체 손 길이
- Hawkeye: 검지 길이 (MCP→PIP→DIP→TIP)

### 4.2 Gait Features (47개)

#### ✅ 상대적 측정 (35개) - 카메라 불변

| Category | Features |
|----------|----------|
| 시간 기반 | `cadence`, `swing_time_mean`, `stance_time_mean`, `double_support_time` |
| 비율 | `swing_stance_ratio`, `double_support_percent` |
| 변동계수 | `stride_variability`, `stride_variability_first/second_half` |
| 비대칭 % | `step_length_asymmetry`, `swing_time_asymmetry`, `arm_swing_asymmetry` |
| 횟수 | `step_count`, `left/right_step_count` |
| 지표 | `festination_index`, `gait_regularity` |
| **각도 (degrees)** | `trunk_flexion_mean/rom`, `hip_flexion_rom_*`, `knee_flexion_rom_*`, `ankle_dorsiflexion_rom_*` |
| 변화율 % | `step_length_trend`, `cadence_trend`, `arm_swing_trend`, `variability_trend`, `step_height_trend` |

#### ⚠️ 절대/추정 측정 (12개) - 주의 필요

| Feature | 단위 | 측정 방법 |
|---------|------|----------|
| `walking_speed` | m/s | 골반 너비 가정 (0.30m) |
| `stride_length` | m | 골반 너비 가정 |
| `left/right_stride_length` | m | 골반 너비 가정 |
| `arm_swing_amplitude_*` | m | world_landmarks (정확) 또는 추정 |
| `step_height_*` | m | world_landmarks 필요 |
| `step_length_first/second_half` | m | 골반 너비 가정 |
| `arm_swing_first/second_half` | m | world_landmarks 또는 추정 |
| `step_height_first/second_half` | m | world_landmarks 필요 |

#### 스케일 추정 코드 (metrics_calculator.py)
```python
# 골반 너비 기반 스케일 추정
hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
avg_hip_width_norm = np.mean(hip_widths)
REAL_HIP_WIDTH = 0.30  # meters (평균 성인 골반 너비 가정!)
scale_factor = REAL_HIP_WIDTH / avg_hip_width_norm
```

---

## 5. 요약

### 5.1 Feature 상대성 요약

| Task | 전체 | 상대적 | 절대/추정 |
|------|------|--------|----------|
| **Finger Tapping** | 27개 | 27개 (100%) ✅ | 0개 |
| **Gait** | 47개 | 35개 (74%) | 12개 (26%) ⚠️ |

### 5.2 권장사항

1. **Finger Tapping**: 모든 feature 사용 가능 (카메라 불변)

2. **Gait**:
   - 상대적 feature 우선 사용 (시간, 비율, 각도)
   - 미터 단위 feature는 world_landmarks 있을 때만 신뢰
   - 없으면 골반 너비 0.30m 가정값 사용 (오차 가능)

3. **ML 학습 시**:
   - Finger Tapping: 27개 feature 전체 사용 OK
   - Gait: 상대적 feature 위주로 사용 권장

---

## 6. 전처리 기법

### 6.1 현재 적용된 전처리 (2024-12-09 업데이트)

| 기법 | 위치 | 설명 |
|------|------|------|
| **Savitzky-Golay Filter** | Finger Tapping | `savgol_filter(distances, 11, 3)` |
| **Savitzky-Golay Filter** | Gait | `savgol_filter(heel_rel_z, 15, 3)` |
| **Linear Interpolation** | Gait | Missing landmarks 보간 (최대 10프레임) |
| **Visibility Threshold** | 전체 | `VISIBILITY_THRESHOLD = 0.5` |
| **Safe Division** | 전체 | `EPSILON = 1e-8` 으로 division by zero 방지 |
| **Percentage Clipping** | 전체 | `[-500%, 500%]` 범위로 outlier 제한 |

### 6.2 수치 안정성 함수

```python
# backend/services/metrics_calculator.py
EPSILON = 1e-8
PERCENTAGE_CLIP_MIN = -500.0
PERCENTAGE_CLIP_MAX = 500.0

def safe_divide(numerator, denominator, default=0.0):
    """Division by zero 방지"""
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator

def clip_percentage(value):
    """Percentage outlier 제한"""
    return np.clip(value, PERCENTAGE_CLIP_MIN, PERCENTAGE_CLIP_MAX)
```

### 6.3 적용된 Feature들

- `velocity_decrement`, `amplitude_decrement`
- `fatigue_rate`, `velocity_slope`, `amplitude_slope`
- `variability_change`, `variability_trend`
- `step_length_asymmetry`, `swing_time_asymmetry`, `arm_swing_asymmetry`
- 모든 `*_trend` features

---

## 7. 관련 파일

| 파일 | 역할 |
|------|------|
| `backend/services/mediapipe_processor.py` | 스켈레톤 추출 |
| `backend/services/metrics_calculator.py` | Feature 계산 + 전처리 |
| `scripts/extract_finger_tapping_features.py` | Finger Tapping feature 추출 |
| `scripts/extract_features.py` | 전체 task feature 추출 |
| `data/processed/features/*_stratified.csv` | 추출된 features |
