# Hand Movement / Leg Agility 데이터 품질 분석

## 목적

Hand Movement (Pearson 0.593)와 Leg Agility (Pearson 0.221)의 성능 차이 원인을 시각적으로 분석합니다.

## 분석 항목

### 1. 통계 분석 (`visualize_hand_leg_quality.py`)

- **MediaPipe Skeleton 품질**
  - Missing/invalid values (zero frames, NaN)
  - 좌표 범위 (X, Y, Z)
  - 움직임 활발도 (movement variance)
  - UPDRS 점수별 움직임 차이

- **Feature 분포**
  - Feature 평균/표준편차 분포
  - Zero-variance features
  - 이상치(outlier) 탐지

- **출력**:
  - `{task}_skeleton_quality.png` - 4개 subplot (movement 분포, score별 움직임, 좌표 scatter, score 분포)
  - `{task}_feature_distribution.png` - 2개 subplot (feature mean/std 분포)

### 2. 비디오 샘플 시각화 (`visualize_video_samples.py`)

- **ROI 확인**
  - 손/다리가 프레임 안에 제대로 들어오는지
  - 카메라 각도, 거리 문제

- **MediaPipe Skeleton Overlay**
  - Hand: 21 landmarks
  - Leg: 6 landmarks (hips, knees, ankles)

- **Score별 샘플 비교**
  - Score 0-4 각각 2개 샘플 추출
  - 30 프레임 montage 생성

- **출력**:
  - `{task}_score{N}_{video_id}.png` - 6x5 montage (각 score별 2개)

## 실행 방법

### 방법 1: 배치 파일 실행 (권장)

```bash
cd C:\Users\YK\tulip\Hawkeye\scripts\analysis
run_analysis.bat
```

### 방법 2: 개별 실행

```bash
# 통계 분석만
python scripts/analysis/visualize_hand_leg_quality.py

# 비디오 시각화만
python scripts/analysis/visualize_video_samples.py
```

## 필요 파일

### 데이터 파일
- `data/hand_movement_train.pkl`
- `data/hand_movement_test.pkl`
- `data/leg_agility_train.pkl`
- `data/leg_agility_test.pkl`

### 비디오 파일 (선택)
- `data/raw/PD4T/PD4T/PD4T/Videos/Hand movements/`
- `data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/`

**Note**: 비디오 파일이 없으면 통계 분석만 실행됩니다.

## 예상 결과

### Hand Movement (Pearson 0.593) 예상
- ✅ Skeleton 품질 양호
- ✅ ROI 내 손 위치 적절
- ✅ Movement variance 충분
- ⚠️ 일부 개선 여지 있음

### Leg Agility (Pearson 0.221) 예상 문제
- ❌ **Skeleton 품질 낮음**: 다리 landmarks 누락 또는 부정확
- ❌ **ROI 문제**: 다리가 프레임 밖으로 나감 (카메라 각도/거리)
- ❌ **Movement variance 낮음**: 움직임이 미세하거나 감지 어려움
- ❌ **샘플 부족 또는 노이즈**: 작은 데이터셋, 라벨링 품질 낮음

## 출력 위치

```
scripts/analysis/output/
├── hand_movement_skeleton_quality.png
├── hand_movement_feature_distribution.png
├── leg_agility_skeleton_quality.png
├── leg_agility_feature_distribution.png
└── video_samples/
    ├── hand_movement_score0_*.png
    ├── hand_movement_score1_*.png
    ├── ...
    ├── leg_agility_score0_*.png
    └── leg_agility_score1_*.png
```

## 다음 단계

분석 결과에 따라:

1. **Leg Agility 데이터 개선**
   - ROI 조정 (카메라 각도/거리)
   - MediaPipe 파라미터 튜닝
   - Outlier 제거

2. **Feature Engineering**
   - Biomechanics-based features 추가
   - Temporal patterns 개선

3. **Generative AI 데이터 증강**
   - Skeleton data augmentation (GAN/Diffusion)
   - 샘플 수 10배 증가

4. **Hand Movement 추가 개선**
   - Pearson 0.593 → 0.65+ 목표
   - Feature selection, hyperparameter tuning
