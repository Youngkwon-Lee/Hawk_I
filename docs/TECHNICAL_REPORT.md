# Hawkeye Technical Report

**Project**: AI-powered Parkinson's Disease Motor Assessment System
**Version**: 1.0
**Date**: 2024-12-09
**Author**: Hawkeye Research Team

---

## Executive Summary

Hawkeye는 비디오 기반 파킨슨병 운동 증상 평가 시스템입니다. MediaPipe를 활용한 스켈레톤 추출, kinematic feature 계산, ML/DL 모델을 통해 MDS-UPDRS 점수(0-4)를 자동으로 예측합니다.

**핵심 성과**:
- Finger Tapping: Within 1 Point Accuracy **96.1%**
- Gait: Within 1 Point Accuracy **94.2%**
- 전체 74개 kinematic features 추출 (27 FT + 47 Gait)

---

## 1. Dataset

### 1.1 PD4T Dataset

| 항목 | 내용 |
|------|------|
| **출처** | PD4T (Parkinson's Disease for Tele-assessment) |
| **Task 종류** | Finger Tapping, Gait, Hand Movement, Leg Agility |
| **레이블** | MDS-UPDRS Part III Score (0-4) |
| **비디오 형식** | MP4, 30fps |

### 1.2 데이터 분할 (Stratified Split)

**중요**: 원본 PD4T에 Data Leakage 발견 → Stratified Split으로 재구성

| Task | Train | Valid | Test | Total |
|------|-------|-------|------|-------|
| **Finger Tapping** | 561 | 120 | 128 | 809 |
| **Gait** | 510 | 109 | 117 | 736 |

### 1.3 Score Distribution

```
Finger Tapping:
  Score 0: 15.2%
  Score 1: 38.4%
  Score 2: 31.7%
  Score 3: 12.1%
  Score 4: 2.6%

Gait:
  Score 0: 12.8%
  Score 1: 35.6%
  Score 2: 33.2%
  Score 3: 14.9%
  Score 4: 3.5%
```

### 1.4 데이터 경로

```
data/
├── raw/PD4T/PD4T/PD4T/
│   ├── Videos/
│   │   ├── Finger tapping/{subject_id}/{video_id}.mp4
│   │   └── Gait/{subject_id}/{video_id}.mp4
│   └── Annotations/
│       ├── Finger tapping/
│       └── Gait/
└── processed/features/
    ├── finger_tapping_*_features_stratified.csv
    └── gait_*_features_stratified.csv
```

---

## 2. Skeleton Extraction

### 2.1 MediaPipe Configuration

| 설정 | Finger Tapping | Gait |
|------|----------------|------|
| **Mode** | Hand | Pose |
| **Landmarks** | 21 points | 33 points |
| **Model Complexity** | 1 | 2 |
| **Min Detection Confidence** | 0.5 | 0.5 |
| **Min Tracking Confidence** | 0.5 | 0.5 |

### 2.2 Hand Landmarks (21개)

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
                    ========0 (WRIST)========================
```

**핵심 포인트**:
- `4` (THUMB_TIP) ↔ `8` (INDEX_TIP): 탭핑 거리 측정
- `5-8` (INDEX finger): 정규화 기준 길이

### 2.3 Pose Landmarks (33개)

```
                    0 (nose)
                   /|\
                1-2 3-4 (eyes, ears)
                   │
               11──┼──12  (shoulders)
              /    │    \
            13    │     14  (elbows)
                   │
            23────┼────24  (hips)
              \   │   /
              25  │  26  (knees)
               \  │  /
              27  │  28  (ankles)
               │     │
              31    32  (foot index)
```

**Gait 분석 핵심 포인트**: 23-32 (하지), 11-16 (상체/팔)

### 2.4 좌표 시스템

| 좌표 | Hand | Pose | 설명 |
|------|------|------|------|
| `x, y` | 0.0~1.0 | 0.0~1.0 | 이미지 정규화 |
| `z` | 상대값 | 상대값 | 손목/골반 기준 깊이 |
| `world_landmarks` | ❌ | ✅ | 실제 미터 단위 |
| `visibility` | ✅ | ✅ | 신뢰도 0.0~1.0 |

---

## 3. Feature Engineering

### 3.1 Finger Tapping Features (27개)

모든 feature는 **카메라 불변** (검지 길이 정규화)

| Category | Features | 설명 |
|----------|----------|------|
| **Speed** | `tapping_speed` | Hz (탭/초) |
| **Amplitude** | `amplitude_mean`, `amplitude_std`, `amplitude_decrement` | 정규화된 탭핑 진폭 |
| **Velocity** | `opening_velocity_mean`, `closing_velocity_mean`, `peak_velocity_mean`, `velocity_decrement` | 정규화된 속도 |
| **Rhythm** | `rhythm_variability`, `rhythm_slope` | CV% 변동성 |
| **Events** | `hesitation_count`, `halt_count`, `freeze_episodes` | 이상 이벤트 카운트 |
| **Fatigue** | `fatigue_rate` | 피로도 (%/tap) |
| **Temporal** | `velocity_first/mid/last_third`, `amplitude_first/mid/last_third` | 시간 구간별 분석 |
| **Trend** | `velocity_slope`, `amplitude_slope`, `variability_change` | 시간에 따른 변화율 |

### 3.2 Gait Features (47개)

| Category | Count | 상대적 | Features |
|----------|-------|--------|----------|
| **Temporal** | 6 | ✅ | `cadence`, `swing_time_mean`, `stance_time_mean`, etc. |
| **Spatial** | 8 | ⚠️ | `stride_length`, `walking_speed` (골반 가정) |
| **Symmetry** | 3 | ✅ | `step_length_asymmetry`, `swing_time_asymmetry`, `arm_swing_asymmetry` |
| **Arm Swing** | 5 | ⚠️ | `arm_swing_amplitude_*` |
| **Joint Angles** | 12 | ✅ | `trunk_flexion_*`, `hip_flexion_rom_*`, `knee_flexion_rom_*`, `ankle_dorsiflexion_rom_*` |
| **Variability** | 4 | ✅ | `stride_variability`, `stride_variability_first/second_half` |
| **Trend** | 9 | ✅ | `step_length_trend`, `cadence_trend`, etc. |

**Feature 상대성 요약**:
- Finger Tapping: 27/27 (100%) 상대적 ✅
- Gait: 35/47 (74%) 상대적, 12/47 (26%) 절대/추정 ⚠️

### 3.3 정규화 방법

**Finger Tapping** (검지 길이 정규화):
```python
# Index finger length = MCP→PIP + PIP→DIP + DIP→TIP
seg1 = np.linalg.norm(index_pip - index_mcp)
seg2 = np.linalg.norm(index_dip - index_pip)
seg3 = np.linalg.norm(index_tip - index_dip)
index_finger_length = seg1 + seg2 + seg3

# Normalize
distances_normalized = distances / np.max(finger_lengths)
```

**Gait** (골반 너비 스케일링):
```python
# Hip width based scaling
hip_widths = np.linalg.norm(right_hip - left_hip, axis=1)
REAL_HIP_WIDTH = 0.30  # meters (assumption)
scale_factor = REAL_HIP_WIDTH / np.mean(hip_widths)
```

### 3.4 연구 레퍼런스

| 출처 | 정규화 방법 | 링크 |
|------|------------|------|
| **VisionMD (2025)** | Hand length | [Nature npj PD](https://www.nature.com/articles/s41531-025-00876-6) |
| **PMC10674854 (2023)** | Palm size | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10674854/) |
| **MDS-UPDRS** | Clinical scoring | Movement Disorder Society |

---

## 4. Preprocessing

### 4.1 Signal Smoothing

| 기법 | 적용 대상 | 파라미터 | 목적 |
|------|----------|----------|------|
| **Savitzky-Golay Filter** | Finger Tapping distances | window=11, order=3 | 노이즈 제거 |
| **Savitzky-Golay Filter** | Gait heel trajectory | window=15, order=3 | 보행 주기 검출 |

### 4.2 Missing Data Handling

| 기법 | 조건 | 설명 |
|------|------|------|
| **Linear Interpolation** | gap ≤ 10 frames | 짧은 누락 구간 보간 |
| **Visibility Threshold** | visibility < 0.5 | 신뢰도 낮은 landmark 제외 |

### 4.3 Numerical Stability (2024-12-09 추가)

**문제**: Division by zero로 인한 overflow (max 9.36e+16)

**해결책**:
```python
EPSILON = 1e-8
PERCENTAGE_CLIP_MIN = -500.0
PERCENTAGE_CLIP_MAX = 500.0

def safe_divide(numerator, denominator, default=0.0):
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator

def clip_percentage(value):
    return np.clip(value, PERCENTAGE_CLIP_MIN, PERCENTAGE_CLIP_MAX)
```

**적용된 Features**:
- `velocity_decrement`, `amplitude_decrement`
- `fatigue_rate`, `velocity_slope`, `amplitude_slope`
- `variability_change`, `variability_trend`
- `step_length_asymmetry`, `swing_time_asymmetry`, `arm_swing_asymmetry`

---

## 5. Models

### 5.1 ML Models

| Model | Task | Test Accuracy | Within 1 | MAE |
|-------|------|---------------|----------|-----|
| **XGBoost Regressor** | Finger Tapping | 58.6% | 96.1% | 0.52 |
| **Random Forest Regressor** | Finger Tapping | 56.3% | 94.5% | 0.55 |
| **XGBoost Regressor** | Gait | 54.7% | 94.2% | 0.58 |
| **Random Forest Regressor** | Gait | 52.1% | 92.8% | 0.61 |

### 5.2 Hyperparameters

**XGBoost**:
```python
XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror',
    eval_metric='mae'
)
```

**Random Forest**:
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1
)
```

### 5.3 Feature Importance (Top 10)

**Finger Tapping**:
1. `rhythm_variability` (0.142)
2. `velocity_decrement` (0.098)
3. `amplitude_decrement` (0.087)
4. `peak_velocity_mean` (0.076)
5. `hesitation_count` (0.065)

**Gait**:
1. `stride_variability` (0.121)
2. `arm_swing_asymmetry` (0.098)
3. `trunk_flexion_mean` (0.087)
4. `cadence` (0.076)
5. `swing_stance_ratio` (0.071)

### 5.4 Model Files

| 파일 | 용도 |
|------|------|
| `models/trained/xgb_finger_tapping_scorer.pkl` | XGBoost FT 모델 |
| `models/trained/rf_finger_tapping_scorer.pkl` | Random Forest FT 모델 |
| `models/trained/xgb_gait_scorer.pkl` | XGBoost Gait 모델 |
| `models/trained/rf_gait_scorer.pkl` | Random Forest Gait 모델 |
| `models/trained/finger_tapping_feature_cols.json` | Feature 컬럼 정의 |

---

## 6. Training Environment

### 6.1 Hardware Requirements

| Component | ML Models | DL Models (LSTM) |
|-----------|-----------|------------------|
| **CPU** | Intel i7+ | Intel i7+ |
| **RAM** | 16GB | 32GB+ |
| **GPU** | Optional | CUDA GPU (RTX 3080+) |
| **Storage** | 50GB | 100GB+ |

### 6.2 Software Stack

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| MediaPipe | 0.10.x |
| PyTorch | 2.0+ |
| XGBoost | 2.0+ |
| scikit-learn | 1.3+ |
| OpenCV | 4.8+ |

### 6.3 Environment Matrix

| 모델 유형 | 로컬 | HPC | Cloud |
|----------|------|-----|-------|
| ML (RF, XGBoost) | ✅ 권장 | ✅ | - |
| DL (LSTM) | ⚠️ 테스트만 | ✅ 권장 | ✅ |
| VLM (Qwen, LLaVA) | ❌ | ✅ 권장 | ✅ API |

### 6.4 Training Commands

```bash
# Finger Tapping ML
python scripts/training/train_finger_tapping_ml.py

# Gait ML
python scripts/training/train_gait_ml.py

# LSTM (HPC)
python scripts/training/train_gait_lstm.py
```

---

## 7. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                 │
│                    Video (MP4, 30fps)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SKELETON EXTRACTION                        │
│                 MediaPipeProcessor                           │
│           ┌─────────────┬─────────────┐                     │
│           │   Hand Mode │  Pose Mode  │                     │
│           │ 21 landmarks│ 33 landmarks│                     │
│           └─────────────┴─────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                             │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Savitzky-Golay │  │ Interpolation│  │ Visibility     │  │
│  │ Smoothing      │  │ (gap ≤ 10)   │  │ Threshold      │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION                          │
│                  MetricsCalculator                           │
│           ┌─────────────┬─────────────┐                     │
│           │   27 FT     │   47 Gait   │                     │
│           │  Features   │  Features   │                     │
│           └─────────────┴─────────────┘                     │
│                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ safe_divide()  │  │ clip_pct()   │  │ Normalization  │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ML PREDICTION                            │
│           ┌─────────────┬─────────────┐                     │
│           │  XGBoost    │   Random    │                     │
│           │  Regressor  │   Forest    │                     │
│           └─────────────┴─────────────┘                     │
│                         │                                    │
│                         ▼                                    │
│               Round & Clip [0-4]                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                                │
│              MDS-UPDRS Score (0-4)                          │
│           + Confidence + Feature Analysis                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key Files Reference

| Category | File | Purpose |
|----------|------|---------|
| **Skeleton** | `backend/services/mediapipe_processor.py` | MediaPipe 스켈레톤 추출 |
| **Features** | `backend/services/metrics_calculator.py` | Feature 계산 + 전처리 |
| **Training** | `scripts/training/train_finger_tapping_ml.py` | FT ML 학습 |
| **Training** | `scripts/training/train_gait_ml.py` | Gait ML 학습 |
| **Extraction** | `scripts/extract_finger_tapping_features.py` | FT feature 추출 |
| **Extraction** | `scripts/extract_features.py` | Gait feature 추출 |
| **Config** | `scripts/config.py` | 경로 설정 |
| **Models** | `models/trained/*.pkl` | 학습된 모델 |
| **Data** | `data/processed/features/*_stratified.csv` | 추출된 features |

---

## 9. Known Issues & Limitations

### 9.1 해결된 이슈

| Issue | 원인 | 해결책 | 날짜 |
|-------|------|--------|------|
| Data Leakage | 원본 PD4T train/test 환자 중복 | Stratified split | 2024-12-09 |
| Feature Overflow | Division by zero | safe_divide + clip | 2024-12-09 |

### 9.2 현재 제한사항

1. **Gait 절대 측정치**: 골반 너비 0.30m 가정 (개인차 존재)
2. **Hand world_landmarks 미지원**: MediaPipe 제한
3. **조명/배경 민감도**: 극단적 조명에서 detection 저하
4. **Score 4 불균형**: 전체의 2-4%로 학습 데이터 부족

### 9.3 향후 개선 방향

1. LSTM/Transformer 기반 sequence 모델 적용
2. VLM (Vision-Language Model) 통합
3. 멀티태스크 학습 (FT + Gait + Hand Movement + Leg Agility)
4. Real-time inference 최적화

---

## 10. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-09 | Initial release |
| 1.0.1 | 2024-12-09 | Data leakage fix (stratified split) |
| 1.0.2 | 2024-12-09 | Numerical overflow fix (safe_divide, clip_percentage) |

---

## Appendix A: Feature Definitions

### A.1 Finger Tapping Features (27)

```json
[
  "tapping_speed",
  "amplitude_mean", "amplitude_std", "amplitude_decrement",
  "first_half_amplitude", "second_half_amplitude",
  "opening_velocity_mean", "closing_velocity_mean", "peak_velocity_mean",
  "velocity_decrement",
  "rhythm_variability",
  "hesitation_count", "halt_count", "freeze_episodes",
  "fatigue_rate",
  "velocity_first_third", "velocity_mid_third", "velocity_last_third",
  "amplitude_first_third", "amplitude_mid_third", "amplitude_last_third",
  "velocity_slope", "amplitude_slope", "rhythm_slope",
  "variability_first_half", "variability_second_half", "variability_change"
]
```

### A.2 Gait Features (47)

See `docs/SKELETON_FEATURES_ANALYSIS.md` for complete list.

---

## Appendix B: MDS-UPDRS Scoring Criteria

| Score | Clinical Description |
|-------|---------------------|
| 0 | Normal |
| 1 | Slight: minimal impairment |
| 2 | Mild: clear impairment |
| 3 | Moderate: considerable impairment |
| 4 | Severe: cannot perform task |

---

**Document End**
