# Leg Agility Task 실패 원인 종합 분석 보고서

**작성일**: 2025-12-24
**분석 기간**: 2025-12-16 ~ 2025-12-24
**결론**: **Leg Agility Task 포기 권장**

---

## Executive Summary

Leg Agility task는 **Pearson correlation 0.221**로 실패했습니다. 8일간의 집중 분석 결과, 근본 원인은:

1. **극심한 클래스 불균형** (Score 3,4 단 10개 샘플)
2. **앉은 자세로 인한 Hip-Knee joint overlap** 문제
3. **Skeleton features가 육안으로 보이는 미묘한 차이를 포착 불가**

**최종 권장사항**: Leg Agility task **포기** 또는 **추가 데이터 수집** 후 재시도

---

## 1. Baseline 성능 비교

| Task | Pearson | MAE | Status |
|------|---------|-----|--------|
| **Gait** | 0.807 | 0.241 | ✅ 성공 |
| **Finger Tapping** | 0.609 | 0.444 | ✅ 성공 |
| **Hand Movement** | 0.593 | 0.477 | ✅ 성공 |
| **Leg Agility** | **0.221** | 0.566 | ❌ **실패** |

Leg Agility는 유일하게 0.3 미만의 상관관계를 보였습니다.

---

## 2. 데이터 품질 분석

### 2.1 클래스 분포 (치명적 불균형)

```
Score 0: 297 samples (48.9%)
Score 1: 262 samples (43.2%)
Score 2:  38 samples ( 6.3%)
Score 3:   7 samples ( 1.2%)  ← 극소수!
Score 4:   3 samples ( 0.5%)  ← 극소수!
```

**문제**: Score 3,4 합쳐서 단 **10개 샘플** (전체의 1.6%)
- Binary classification도 불가능 (test set에 severe 1개뿐)
- 통계적 신뢰성 전무

### 2.2 Movement Variance Analysis

**결과**: Score와 movement magnitude 간 **상관관계 없음**

```
Score 0: Movement = 0.042
Score 1: Movement = 0.044
Score 2: Movement = 0.041
Score 3: Movement = 0.040
Score 4: Movement = 0.038
```

**다른 task와 비교**:
- Hand Movement: Score 0 (0.063) → Score 4 (0.075) ✅ 증가 패턴
- Leg Agility: 모든 score ~0.04 ❌ 패턴 없음

### 2.3 ROI (Region of Interest) 분석

**Hand Movement**:
- Landmarks가 중앙에 집중 (X: 0.4-0.6, Y: 0.4-0.6)
- 일관된 카메라 위치

**Leg Agility**:
- Landmarks가 전체 영역에 산재 (X: 0.0-1.0, Y: 0.0-1.0)
- 비일관적 카메라 설정 또는 피험자 위치

---

## 3. Hip-Knee Joint Overlap 문제

### 3.1 문제 발견

**육안 관찰**: Score 4 영상에서 hip과 knee가 거의 겹쳐 보임

**정량 분석**:
```
Score 0: Hip-Knee distance = 0.161 (16cm)
Score 1: Hip-Knee distance = 0.159
Score 2: Hip-Knee distance = 0.160
Score 3: Hip-Knee distance = 0.123
Score 4: Hip-Knee distance = 0.036 (3.6cm!) ← 거의 붙음!
```

### 3.2 원인

**Task 특성**: "앉아있는 상태에서 hip flexion 반복"
- 앉은 자세 → hip joint가 상대적으로 고정
- Score 4 환자는 다리를 거의 못 올림 → hip과 knee가 근접

### 3.3 해결 시도 1: Full Body Landmarks

**제안**: MediaPipe Pose 33개 전체 사용 (upper body 포함)
- 상체 움직임(torso, shoulders)으로 보상 동작 감지

**문제**: Full body를 써도 **hip overlap 문제는 해결 안 됨**
- Hip이 어차피 앉아있어서 고정됨
- Upper body 추가는 noise만 증가

### 3.4 해결 시도 2: Knee-Only Approach (Hip 제거)

**전략**:
- Hip landmarks (0,1) 제거
- Knee (2,3) + Ankle (4,5)만 사용 (4 landmarks)
- Hip-Knee overlap 문제 회피

**결과**: Scripts analysis 섹션 참조

---

## 4. Feature Engineering 시도

### 4.1 Knee Trajectory Features (11개)

**기본 Features**:
1. `knee_range_y`: 무릎 상하 움직임 범위
2. `mean_velocity`: 평균 속도
3. `max_velocity`: 최대 속도
4. `mean_accel`: 평균 가속도
5. `num_peaks`: 움직임 주기 수
6. `rhythm_std`: 리듬 일정성 (변동성)
7. `knee_ankle_dist`: 무릎-발목 거리 (무릎 굽힘 정도)

**Parkinson-Specific Features** (추가):
8. `early_velocity`: 초기 30 프레임 속도
9. `late_velocity`: 말기 30 프레임 속도
10. `velocity_decrement`: 초기 대비 말기 속도 감소율 (fatigue)
11. `amplitude_decrement`: 초기 대비 말기 진폭 감소율 (bradykinesia)

**근거**: 파킨슨병 특징 = "속도, 타이밍, 일관성, 초기→말기 변화"

**결과 (참담)**:
```
knee_range_y        : Pearson = -0.064
mean_velocity       : Pearson = -0.064
max_velocity        : Pearson = -0.005
mean_accel          : Pearson = -0.027
num_peaks           : Pearson = -0.000
rhythm_std          : Pearson = -0.039
knee_ankle_dist     : Pearson = -0.059
early_velocity      : Pearson = -0.064
late_velocity       : Pearson = -0.021
velocity_decrement  : Pearson = +0.035  ← 파킨슨 핵심, but 무의미
amplitude_decrement : Pearson = +0.029  ← 파킨슨 핵심, but 무의미
```

**모든 features < 0.1**, 전혀 쓸모없음.

### 4.2 Jerk & Smoothness Features (최종 시도)

**동기**: 육안으로 관찰된 "부자연스러움, 부드럽움 떨어짐, 힘이 들어간 느낌"

**Features**:
1. `mean_jerk`: 평균 jerk (3차 미분)
2. `max_jerk`: 최대 jerk
3. `jerk_std`: Jerk 변동성
4. `smoothness`: -log(mean_jerk) - 부드러움 지표
5. `accel_variance`: 가속도 분산 (경직성)
6. `movement_efficiency`: 경로 길이 / 직선 거리

**결과 (최악)**:
```
mean_jerk           : Pearson = -0.018
max_jerk            : Pearson = -0.004
jerk_std            : Pearson = -0.009
smoothness          : Pearson = +0.051  ← 가장 높지만 여전히 무의미
accel_variance      : Pearson = -0.012
movement_efficiency : Pearson = -0.007
```

**결론**: **Skeleton으로는 육안 차이 포착 불가**

---

## 5. Binary Classification 시도

### 5.1 동기

사용자 피드백:
> "3점과 4점은 누가봐도 rulebased로 잡을 정도"
> "그런데 ml이나 통계로 3,4점과 0,1,2점도 구분못하면 문제가 많아"

→ 최소한 Severe (3,4) vs Mild (0,1,2) 이진 분류는 되어야 함

### 5.2 결과

**겉보기 성능**:
- Logistic Regression: Accuracy = 99.2%
- Random Forest: Accuracy = 99.2%

**실제 Confusion Matrix**:
```
              Predicted
              Mild  Severe
Actual Mild   121     0
Actual Severe   1     0  ← 100% 실패!
```

**분석**:
- 모델이 **모든 샘플을 Mild로 예측** (majority baseline)
- Severe class: Precision = 0.0, Recall = 0.0
- Baseline 대비 improvement = 0.000
- **완전 실패**

### 5.3 실패 원인

**극심한 클래스 불균형**:
- Mild (0,1,2): 597 samples (98.4%)
- Severe (3,4): 10 samples (1.6%)

Test set에 severe 샘플이 **단 1개**만 있어 통계적 검증 불가능.

---

## 6. 육안 분석 vs Skeleton Features

### 6.1 육안으로 관찰된 차이

**Score 4**:
- "아예 발을 못 떼네"
- 무릎이 거의 움직이지 않음
- Hip-Knee distance = 0.036 (3.6cm)

**Score 3**:
- "무릎을 높게 못들고"
- "빠르게 tapping을 못하고"
- "tapping도 불규칙하고"

**Score 2**:
- "약간 부자연스러워"
- "부드럽움이 떨어지고 힘이들어간 느낌"
- "tapping이 빠르긴한데 불규칙적이고"
- "무릎높이가 다소 낮아"

### 6.2 Skeleton Features로 포착 가능 여부

| 육안 관찰 | 대응 Feature | Pearson | 포착 여부 |
|----------|-------------|---------|----------|
| "발을 못 뗌" | knee_range_y | -0.064 | ❌ |
| "빠르게 못함" | mean_velocity | -0.064 | ❌ |
| "불규칙" | rhythm_std | -0.039 | ❌ |
| "부자연스러움" | smoothness | +0.051 | ❌ |
| "힘이 들어감" | accel_variance | -0.012 | ❌ |
| "초기→말기 감소" | velocity_decrement | +0.035 | ❌ |

**결론**: **육안으로 보이는 모든 차이를 skeleton features로 포착 불가**

---

## 7. 근본 원인 분석

### 7.1 가능한 원인

#### A. Skeleton 추출 품질 문제
- **증거**:
  - ROI 산재 (일관성 없는 카메라 위치)
  - Hip-Knee overlap (MediaPipe가 정확히 구분 못함)
  - 앉은 자세에서 landmark 추출 어려움

- **대안**:
  - 다른 pose estimation 모델 시도 (OpenPose, AlphaPose)
  - 카메라 각도 통일된 데이터 재수집

#### B. Labeling 품질 문제
- **증거**:
  - 모든 features가 score와 무관 (Pearson ~0)
  - Binary classification도 실패
  - 육안 관찰과 feature 값 불일치

- **대안**:
  - UPDRS 재평가 (다수 전문가 합의)
  - Inter-rater reliability 측정
  - 라벨링 가이드라인 재검토

#### C. 데이터 부족 문제
- **증거**:
  - Score 3,4 단 10개 샘플 (1.6%)
  - Binary classification 불가능
  - 통계적 검정력 부족

- **대안**:
  - Score 3,4 샘플 대규모 추가 수집 (최소 100개 이상)
  - Data augmentation (temporal jittering, rotation)
  - Synthetic data generation

#### D. Task 특성 문제
- **증거**:
  - 앉은 자세 → hip 고정 → 정보 손실
  - Full body로도 해결 안 됨
  - 다른 task(Gait, Finger)는 성공

- **대안**:
  - Standing position으로 task 재정의
  - 측면 카메라 추가 (sagittal plane)
  - Force plate 등 추가 센서 사용

### 7.2 가장 유력한 원인

**1순위: 데이터 부족 + 클래스 불균형**
- Score 3,4가 10개뿐이라 ML 자체가 불가능
- 이진 분류조차 test set에 1개뿐

**2순위: Task 특성 (앉은 자세)**
- Hip-Knee overlap은 구조적 문제
- MediaPipe 2D skeleton으로 해결 불가

**3순위: Labeling 품질**
- 육안 차이는 명확히 보이는데 features와 불일치
- Score 2-4 간 차이가 미묘할 가능성

---

## 8. 다른 Task 성공 사례와 비교

### 8.1 성공한 Task 특징

#### Gait (Pearson 0.807)
- **서있는 자세** → hip, knee, ankle 모두 자유롭게 움직임
- 충분한 클래스 분포
- 명확한 움직임 패턴 (stride, cadence)

#### Finger Tapping (Pearson 0.609)
- **손가락** → 21개 landmarks (정밀한 추적)
- 명확한 tapping 주기
- ROI 일관성 (손은 항상 중앙)

#### Hand Movement (Pearson 0.593)
- **손** → 21개 landmarks
- Opening/closing 명확
- ROI 일관성

### 8.2 Leg Agility와의 차이

| 특징 | 성공 Task | Leg Agility |
|------|----------|-------------|
| **자세** | 서있거나 손만 | **앉은 자세** (hip 고정) |
| **Landmarks** | 21개 (손) | 6개 (다리) |
| **ROI** | 일관적 | 산재 |
| **클래스 분포** | 균형적 | 극심한 불균형 (3,4 = 1.6%) |
| **움직임 명확성** | 명확 | 미묘 |

---

## 9. 최종 결론

### 9.1 포기 권장 근거

1. **통계적 검정력 부족**
   - Score 3,4 단 10개 → binary classification 불가능
   - 추가 데이터 없이는 ML 자체가 불가능

2. **Feature 완전 실패**
   - 11개 knee features: 모두 Pearson < 0.1
   - 6개 jerk/smoothness features: 모두 Pearson < 0.1
   - 총 17개 features 모두 무의미

3. **구조적 문제**
   - Hip-Knee overlap (앉은 자세의 본질적 한계)
   - MediaPipe 2D로 해결 불가

4. **시간 대비 효과**
   - 8일 집중 분석에도 돌파구 없음
   - 추가 feature engineering도 효과 없을 것으로 예상

### 9.2 재시도 조건

만약 재도전한다면:

1. **데이터 수집** (필수):
   - Score 3,4 최소 **100개 이상** 수집
   - 균형잡힌 클래스 분포 (각 20% 이상)

2. **Task 재정의** (권장):
   - Standing position으로 변경
   - 측면 카메라 추가 (sagittal plane)
   - Force plate/IMU 등 추가 센서

3. **Labeling 재검토** (권장):
   - 다수 전문가 합의
   - Inter-rater reliability 0.7 이상 확보
   - 명확한 평가 기준 수립

4. **모델 변경** (선택):
   - 3D pose estimation (MediaPipe 3D, OpenPose)
   - Video-based models (I3D, SlowFast)
   - Multi-modal (skeleton + RGB + depth)

### 9.3 권장사항

**Short-term (현재)**:
- ✅ Leg Agility task **포기**
- ✅ Gait, Finger, Hand 3개 task에 집중
- ✅ 논문/보고서에 실패 사례로 기록 (투명성)

**Long-term (향후)**:
- ⏳ Score 3,4 샘플 100개 이상 수집 후 재도전
- ⏳ Standing position task로 재정의 고려
- ⏳ 3D pose estimation 또는 multi-modal 접근

---

## 10. Lessons Learned

1. **클래스 불균형은 모든 것을 망친다**
   - Score 3,4 1.6% → binary classification도 불가능
   - Feature engineering 이전에 데이터 분포 확인 필수

2. **Task 특성이 모델 성능을 좌우한다**
   - 앉은 자세 → hip 고정 → 정보 손실
   - Standing position의 Gait가 성공한 이유

3. **육안 관찰 ≠ Skeleton features**
   - "부드럽움", "힘이 들어감"은 skeleton으로 포착 어려움
   - RGB video 또는 depth 센서 필요

4. **조기 포기도 중요한 결정이다**
   - 8일 분석에도 돌파구 없으면 포기
   - 리소스를 성공 가능한 task에 집중

---

## 11. Appendix: Analysis Scripts

### Scripts Created

1. `scripts/analysis/visualize_hand_leg_quality.py`
   - Statistical analysis (movement variance, ROI)

2. `scripts/analysis/visualize_skeleton_from_pkl.py`
   - Skeleton sequence visualization

3. `scripts/analysis/compare_leg_landmarks.py`
   - Hip-Knee overlap analysis

4. `scripts/analysis/test_knee_only_approach.py`
   - Knee-only features (11 features)

5. `scripts/analysis/test_jerk_smoothness.py`
   - Jerk/smoothness features (6 features)

6. `scripts/analysis/test_binary_classification.py`
   - Binary classification (severe vs mild)

### Outputs Generated

- `leg_agility_skeleton_quality.png` - Movement variance by score
- `leg_joint_overlap_analysis.png` - Hip-Knee distance visualization
- `leg_why_fullbody_needed.png` - Problem explanation diagram
- `leg_knee_trajectory_features.png` - Knee features by score
- `leg_approach_comparison.png` - Original vs knee-only
- `leg_jerk_smoothness_features.png` - Jerk features by score

---

## 12. References

### PD4T Dataset
- Videos: `data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/`
- Annotations: `data/raw/PD4T/PD4T/PD4T/Annotations/Leg Agility/`

### Feature Files
- `data/leg_agility_train.pkl` (N=485, T=150, J=6, C=3)
- `data/leg_agility_test.pkl` (N=122, T=150, J=6, C=3)

### MediaPipe Landmarks (6 landmarks)
```
0: Left Hip
1: Right Hip
2: Left Knee
3: Right Knee
4: Left Ankle
5: Right Ankle
```

---

**문서 종료**

**최종 권장**: **Leg Agility Task 포기**, Gait/Finger/Hand 3개 task에 집중
