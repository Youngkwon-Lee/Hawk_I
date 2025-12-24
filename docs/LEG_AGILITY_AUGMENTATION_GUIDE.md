# Leg Agility Score 3,4 Data Augmentation Guide

**작성일**: 2025-12-24
**목적**: Score 3,4 (10개 샘플) → 100개 이상으로 증강

---

## Executive Summary

**현재 상황**:
- Score 3,4: 단 10개 샘플 (1.6%)
- 0,1,2점도 구분 불가 (accuracy 50.8% = baseline 51.7%보다 낮음)
- Binary classification도 실패 (severe 완전히 예측 못함)

**증강 필요성**:
- 최소 100개 이상 필요 (class당 10% 이상)
- 10배 증강 필요 (10 → 100+)

**권장 방법**:
1. ✅ **Temporal Augmentation** (가장 안전, 효과적)
2. ✅ **Spatial Augmentation** (주의 필요, 중간 효과)
3. ⚠️ **SMOTE** (feature space, 조건부 사용)
4. ❌ **Noise Injection** (위험, 비추천)
5. ❌ **GAN** (데이터 부족으로 불가능)

---

## 1. Temporal Augmentation (시간 도메인)

### 1.1 Time Stretching/Compression

**원리**: 전체 시퀀스의 속도를 변경 (빠르게/느리게)

**구현**:
```python
def time_stretch(sequence, rate=1.2):
    """
    sequence: (T, J, 3) - T frames, J joints, 3 coords
    rate: >1.0 = faster (compression), <1.0 = slower (stretching)
    """
    T, J, C = sequence.shape
    new_T = int(T / rate)

    # Linear interpolation
    old_indices = np.linspace(0, T-1, T)
    new_indices = np.linspace(0, T-1, new_T)

    stretched = np.zeros((new_T, J, C))
    for j in range(J):
        for c in range(C):
            stretched[:, j, c] = np.interp(new_indices, old_indices, sequence[:, j, c])

    # Pad or truncate to original length
    if new_T < T:
        # Pad with last frame
        padding = np.repeat(stretched[-1:], T - new_T, axis=0)
        stretched = np.concatenate([stretched, padding], axis=0)
    else:
        # Truncate
        stretched = stretched[:T]

    return stretched
```

**증강 배수**: 3-5배
- rate = 0.8 (20% slower)
- rate = 0.9 (10% slower)
- rate = 1.1 (10% faster)
- rate = 1.2 (20% faster)

**장점**:
- ✅ 안전 (biomechanical constraints 유지)
- ✅ 의미 보존 (같은 동작, 다른 속도)
- ✅ 파킨슨 특성 반영 (bradykinesia = 느린 움직임)

**단점**:
- ⚠️ 너무 빠르게/느리게 하면 비현실적

**권장 범위**:
- Score 3: 0.85 - 1.15 (±15%)
- Score 4: 0.9 - 1.1 (±10%, 이미 매우 느림)

---

### 1.2 Temporal Cropping (Window Sliding)

**원리**: 150프레임 시퀀스에서 다른 시작점 선택

**구현**:
```python
def temporal_crop(sequence, crop_length=150, num_crops=5):
    """
    sequence: (T, J, 3) - T >= crop_length
    """
    T, J, C = sequence.shape

    if T < crop_length:
        # Pad if needed
        padding = np.repeat(sequence[-1:], crop_length - T, axis=0)
        sequence = np.concatenate([sequence, padding], axis=0)
        T = crop_length

    # Generate random start points
    crops = []
    max_start = T - crop_length

    if max_start <= 0:
        # Only one crop possible
        return [sequence[:crop_length]]

    for i in range(num_crops):
        start = np.random.randint(0, max_start + 1)
        crop = sequence[start:start + crop_length]
        crops.append(crop)

    return crops
```

**증강 배수**: 3-5배

**장점**:
- ✅ 완전히 안전 (원본 데이터의 일부)
- ✅ Overfitting 방지
- ✅ 다른 movement phase 학습

**단점**:
- ⚠️ 원본이 150프레임보다 길어야 함
- ⚠️ 정보 손실 가능 (중요한 부분 잘릴 수 있음)

**권장**:
- 원본 비디오가 150프레임보다 긴 경우만 사용
- Overlapping crops (50% overlap)

---

### 1.3 Frame Sampling (Subsampling)

**원리**: 프레임을 다르게 샘플링 (매 2프레임, 매 3프레임)

**구현**:
```python
def frame_subsample(sequence, stride=2):
    """
    sequence: (T, J, 3)
    stride: 2 = 매 2프레임마다 선택
    """
    T, J, C = sequence.shape

    # Subsample
    subsampled = sequence[::stride]

    # Interpolate back to original length
    sub_T = len(subsampled)
    old_indices = np.linspace(0, T-1, sub_T)
    new_indices = np.linspace(0, T-1, T)

    interpolated = np.zeros((T, J, C))
    for j in range(J):
        for c in range(C):
            interpolated[:, j, c] = np.interp(new_indices, old_indices, subsampled[:, j, c])

    return interpolated
```

**증강 배수**: 2-3배
- stride = 2
- stride = 3

**장점**:
- ✅ 안전
- ✅ 노이즈 감소 효과 (smoothing)

**단점**:
- ⚠️ 고주파 정보 손실
- ⚠️ 빠른 움직임 놓칠 수 있음

---

## 2. Spatial Augmentation (공간 도메인)

### 2.1 Rotation (회전)

**원리**: Skeleton을 작은 각도로 회전

**구현**:
```python
def rotate_skeleton(sequence, angle_deg=5):
    """
    sequence: (T, J, 3) - only use X, Y (ignore Z)
    angle_deg: rotation angle in degrees
    """
    T, J, C = sequence.shape
    angle_rad = np.deg2rad(angle_deg)

    # Rotation matrix (2D, around Z-axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotated = sequence.copy()
    for t in range(T):
        for j in range(J):
            x, y = sequence[t, j, 0], sequence[t, j, 1]

            # Rotate around center (0.5, 0.5)
            x_centered = x - 0.5
            y_centered = y - 0.5

            x_rot = x_centered * cos_a - y_centered * sin_a
            y_rot = x_centered * sin_a + y_centered * cos_a

            rotated[t, j, 0] = x_rot + 0.5
            rotated[t, j, 1] = y_rot + 0.5

    return rotated
```

**증강 배수**: 4배
- angle = -5°
- angle = -2.5°
- angle = +2.5°
- angle = +5°

**장점**:
- ✅ Camera angle variation 시뮬레이션
- ✅ Robustness 향상

**단점**:
- ⚠️ 너무 크면 비현실적
- ⚠️ Sitting position에서는 제한적 (hip 고정)

**권장 범위**: ±5° 이내

---

### 2.2 Translation (평행 이동)

**원리**: Skeleton을 X, Y 방향으로 이동

**구현**:
```python
def translate_skeleton(sequence, shift_x=0.05, shift_y=0.05):
    """
    sequence: (T, J, 3)
    shift_x, shift_y: translation in normalized coords
    """
    translated = sequence.copy()
    translated[:, :, 0] += shift_x  # X
    translated[:, :, 1] += shift_y  # Y

    # Clip to [0, 1]
    translated[:, :, :2] = np.clip(translated[:, :, :2], 0, 1)

    return translated
```

**증강 배수**: 4배
- shift = (-0.05, 0)
- shift = (+0.05, 0)
- shift = (0, -0.05)
- shift = (0, +0.05)

**장점**:
- ✅ Camera position variation
- ✅ 안전

**단점**:
- ⚠️ ROI 벗어나면 clipping 발생
- ⚠️ 정보 손실 가능

---

### 2.3 Scaling (크기 변경)

**원리**: Skeleton 크기 변경 (피험자 키 차이 시뮬레이션)

**구현**:
```python
def scale_skeleton(sequence, scale=1.1):
    """
    sequence: (T, J, 3)
    scale: >1.0 = larger, <1.0 = smaller
    """
    T, J, C = sequence.shape

    # Calculate center of mass
    center = sequence[:, :, :2].mean(axis=1, keepdims=True)  # (T, 1, 2)

    # Scale around center
    scaled = sequence.copy()
    scaled[:, :, :2] = center + (sequence[:, :, :2] - center) * scale

    # Clip to [0, 1]
    scaled[:, :, :2] = np.clip(scaled[:, :, :2], 0, 1)

    return scaled
```

**증강 배수**: 2-3배
- scale = 0.9 (10% smaller)
- scale = 1.1 (10% larger)

**장점**:
- ✅ 피험자 체격 차이 시뮬레이션
- ✅ Depth variation 시뮬레이션

**단점**:
- ⚠️ Hip-Knee overlap 문제 악화 가능
- ⚠️ ROI 벗어날 수 있음

---

### 2.4 Horizontal Flip (좌우 반전)

**원리**: Left ↔ Right 교환

**구현**:
```python
def horizontal_flip(sequence):
    """
    sequence: (T, J, 3)
    J = 6: [Left Hip, Right Hip, Left Knee, Right Knee, Left Ankle, Right Ankle]
    """
    T, J, C = sequence.shape

    flipped = sequence.copy()

    # Flip X coordinate
    flipped[:, :, 0] = 1.0 - sequence[:, :, 0]

    # Swap Left ↔ Right
    # Left Hip (0) ↔ Right Hip (1)
    # Left Knee (2) ↔ Right Knee (3)
    # Left Ankle (4) ↔ Right Ankle (5)
    flipped[:, [0, 1]] = flipped[:, [1, 0]]
    flipped[:, [2, 3]] = flipped[:, [3, 2]]
    flipped[:, [4, 5]] = flipped[:, [5, 4]]

    return flipped
```

**증강 배수**: 2배

**장점**:
- ✅ 완전히 안전 (해부학적으로 유효)
- ✅ Left/Right dominance 무효화

**단점**:
- ⚠️ 파킨슨병은 편측성 (unilateral) 특징이 있을 수 있음
- ⚠️ 원본 비디오가 이미 left/right 구분 없으면 무의미

**권장**: ✅ 반드시 적용

---

## 3. SMOTE (Feature Space Augmentation)

### 3.1 SMOTE 원리

**Synthetic Minority Over-sampling Technique**
- Feature space에서 minority class 합성 샘플 생성
- K-nearest neighbors 사이를 interpolation

**구현**:
```python
from imblearn.over_sampling import SMOTE

# Extract features first
features = extract_all_features(X)  # (N, 14)

# SMOTE
smote = SMOTE(sampling_strategy='minority', k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, y)
```

**증강 배수**: 무제한 (원하는 만큼)

**장점**:
- ✅ Feature space에서 직접 생성
- ✅ Class imbalance 완전 해결
- ✅ 빠름

**단점**:
- ❌ **Skeleton sequence를 재구성 못함**
- ❌ Feature만 증강 (원본 skeleton 없음)
- ❌ Overfitting 위험 (K=3이면 3개 샘플만 사용)

**사용 조건**:
- ⚠️ Feature-based 모델에만 사용 (Random Forest, XGBoost)
- ❌ Deep Learning (LSTM, Transformer)에는 불가능

**권장**:
- Score 3,4가 최소 5개 이상일 때만 사용
- k_neighbors = min(3, n_samples - 1)

---

## 4. Noise Injection (비추천)

### 4.1 Gaussian Noise

**원리**: Landmark 좌표에 작은 Gaussian noise 추가

**구현**:
```python
def add_gaussian_noise(sequence, std=0.01):
    """
    sequence: (T, J, 3)
    std: standard deviation of noise
    """
    noise = np.random.normal(0, std, sequence.shape)
    noisy = sequence + noise

    # Clip to [0, 1]
    noisy[:, :, :2] = np.clip(noisy[:, :, :2], 0, 1)

    return noisy
```

**증강 배수**: 3-5배

**장점**:
- ✅ MediaPipe extraction error 시뮬레이션
- ✅ Robustness 향상

**단점**:
- ❌ Biomechanical constraints 위반 가능
- ❌ Hip-Knee overlap 문제 악화
- ❌ 의미 없는 샘플 생성 위험

**비추천 이유**:
- Leg Agility는 이미 skeleton quality가 낮음
- Noise 추가하면 더 나빠짐

---

### 4.2 Landmark Dropout (비추천)

**원리**: 일부 landmark를 random하게 masking

**구현**:
```python
def landmark_dropout(sequence, dropout_rate=0.1):
    """
    sequence: (T, J, 3)
    dropout_rate: probability of dropping each landmark
    """
    T, J, C = sequence.shape

    # Random mask
    mask = np.random.rand(T, J) > dropout_rate  # (T, J)

    dropped = sequence.copy()
    for t in range(T):
        for j in range(J):
            if not mask[t, j]:
                # Set to 0 (missing)
                dropped[t, j, :] = 0

    return dropped
```

**비추천 이유**:
- ❌ Leg Agility는 단 6개 landmarks (매우 적음)
- ❌ 하나라도 dropout하면 정보 심각하게 손실
- ❌ Hip-Knee overlap 이미 문제인데 더 악화

---

## 5. GAN (Generative Adversarial Network) - 불가능

### 5.1 TimeGAN / C-RNN-GAN

**원리**: GAN으로 시계열 skeleton 생성

**왜 불가능한가**:
- ❌ Score 3,4 단 10개로는 GAN 학습 불가능
- ❌ 최소 100-1000개 필요
- ❌ Mode collapse (같은 샘플만 생성)

**대안**:
- Pre-trained GAN 사용? → Leg Agility 특화 GAN 없음
- Transfer learning? → Source domain 없음

---

## 6. 증강 전략 (종합)

### 6.1 추천 Pipeline (10 → 100+ samples)

**Step 1: Horizontal Flip (2배)**
```python
# Score 3: 7개 → 14개
# Score 4: 3개 → 6개
# Total: 10 → 20
```

**Step 2: Time Stretching (5배)**
```python
rates = [0.85, 0.9, 1.0, 1.1, 1.15]
# Total: 20 → 100
```

**Step 3: Rotation (4배, 선택적)**
```python
angles = [-5, -2.5, +2.5, +5]
# Total: 100 → 400 (과도할 수 있음)
```

**Step 4: Translation (2배, 선택적)**
```python
shifts = [(-0.05, 0), (+0.05, 0)]
# Total: 400 → 800 (너무 많음)
```

**권장 조합**:
- **Conservative (보수적)**: Flip + Time Stretch = 10배 (100개)
- **Moderate (중도)**: Flip + Time Stretch + Rotation (±5° only) = 20배 (200개)
- **Aggressive (공격적)**: 모두 적용 = 40배+ (400개+)

---

### 6.2 구현 예시

```python
def augment_leg_agility_score34(sequences, labels, target_count=100):
    """
    Augment Score 3,4 samples to target_count

    sequences: (N, T, J, 3) - N = 10 (Score 3,4 only)
    labels: (N,)
    target_count: target number of samples (e.g., 100)
    """
    augmented_sequences = []
    augmented_labels = []

    # Original samples
    augmented_sequences.extend(sequences)
    augmented_labels.extend(labels)

    # Step 1: Horizontal Flip (2배)
    for seq, label in zip(sequences, labels):
        flipped = horizontal_flip(seq)
        augmented_sequences.append(flipped)
        augmented_labels.append(label)

    # Step 2: Time Stretching (5배, on both original and flipped)
    rates = [0.85, 0.9, 1.1, 1.15]
    current_sequences = augmented_sequences.copy()

    for seq, label in zip(current_sequences, augmented_labels.copy()):
        for rate in rates:
            stretched = time_stretch(seq, rate)
            augmented_sequences.append(stretched)
            augmented_labels.append(label)

    # Step 3: Rotation (optional, 2배)
    if len(augmented_sequences) < target_count:
        angles = [-5, +5]
        current_sequences = augmented_sequences.copy()

        for seq, label in zip(current_sequences, augmented_labels.copy()):
            for angle in angles:
                rotated = rotate_skeleton(seq, angle)
                augmented_sequences.append(rotated)
                augmented_labels.append(label)

    # Shuffle
    augmented_sequences = np.array(augmented_sequences)
    augmented_labels = np.array(augmented_labels)

    indices = np.arange(len(augmented_sequences))
    np.random.shuffle(indices)

    augmented_sequences = augmented_sequences[indices]
    augmented_labels = augmented_labels[indices]

    # Truncate to target_count
    if len(augmented_sequences) > target_count:
        augmented_sequences = augmented_sequences[:target_count]
        augmented_labels = augmented_labels[:target_count]

    print(f"Augmented: {len(sequences)} → {len(augmented_sequences)} samples")

    return augmented_sequences, augmented_labels
```

---

### 6.3 검증 방법

**Before Augmentation**:
```python
# Original: 10 samples
X_score34 = X[(y >= 3)]  # (10, 150, 6, 3)
y_score34 = y[(y >= 3)]  # (10,)
```

**After Augmentation**:
```python
# Augmented: 100 samples
X_aug, y_aug = augment_leg_agility_score34(X_score34, y_score34, target_count=100)
print(X_aug.shape)  # (100, 150, 6, 3)
```

**Sanity Check**:
1. **Shape**: (100, 150, 6, 3) 확인
2. **Range**: X, Y ∈ [0, 1] 확인
3. **Label**: y ∈ {3, 4} 확인
4. **Biomechanics**: Hip-Knee distance > 0 확인
5. **Visualization**: 몇 개 샘플 시각화하여 비현실적이지 않은지 확인

---

## 7. 증강 후 재학습 계획

### 7.1 새로운 데이터셋

**Before**:
```
Score 0: 297 samples (48.9%)
Score 1: 262 samples (43.2%)
Score 2:  38 samples ( 6.3%)
Score 3:   7 samples ( 1.2%)
Score 4:   3 samples ( 0.5%)
Total: 607 samples
```

**After (Conservative)**:
```
Score 0: 297 samples (48.9%)
Score 1: 262 samples (43.2%)
Score 2:  38 samples ( 6.3%)
Score 3,4: 100 samples (14.7%) ← 증강!
Total: 697 samples
```

**After (Moderate)**:
```
Score 0: 297 samples
Score 1: 262 samples
Score 2:  38 samples
Score 3,4: 200 samples (25.1%) ← 증강!
Total: 797 samples
```

### 7.2 재학습 전략

**Option 1: Binary Model (Severe vs Mild)**
```python
# Mild (0,1,2) vs Severe (3,4)
y_binary = (y >= 3).astype(int)

# After augmentation:
# Mild: 597 (75%)
# Severe: 100 (25%) ← 훨씬 나음!
```

**Option 2: 3-Class Model (Mild, Moderate, Severe)**
```python
# Mild (0,1), Moderate (2), Severe (3,4)
y_3class = np.where(y <= 1, 0,  # Mild
                    np.where(y == 2, 1,  # Moderate
                             2))  # Severe
```

**Option 3: 5-Class Ordinal Regression**
```python
# Keep original scores 0,1,2,3,4
# Use ordinal regression (e.g., CORAL)
```

---

## 8. 주의사항

### 8.1 Overfitting 위험

**문제**:
- 증강된 샘플들이 서로 너무 유사
- 원본 10개를 memorize할 위험

**해결**:
1. **Train/Val/Test Split 엄격히**:
   - 원본 10개를 먼저 split
   - 각 split 내에서 증강
   - 절대 augmented sample이 train/test 걸쳐있으면 안 됨

2. **Cross-Validation**:
   - 5-fold CV로 검증
   - 각 fold마다 augmentation 다르게

3. **Regularization**:
   - L2 regularization
   - Dropout (DL 모델)
   - Early stopping

---

### 8.2 증강 비율

**너무 많으면**:
- ❌ Synthetic data가 dominant
- ❌ Original distribution 왜곡
- ❌ Model이 증강 artifact 학습

**너무 적으면**:
- ❌ Class imbalance 해결 안 됨
- ❌ 여전히 학습 불가능

**권장 비율**:
- Score 3,4: 최소 10%, 이상적 20%
- 100개 (14.7%) 또는 200개 (25.1%)

---

## 9. 최종 권장사항

### 9.1 우선순위

**Tier 1 (필수)**:
1. ✅ **Horizontal Flip** - 완전히 안전, 2배
2. ✅ **Time Stretching (±10-15%)** - 안전, 5배

**Tier 2 (권장)**:
3. ✅ **Rotation (±5°)** - 비교적 안전, 2배
4. ⚠️ **Translation (±5%)** - 조건부, 2배

**Tier 3 (선택)**:
5. ⚠️ **Scaling (±10%)** - 주의 필요
6. ⚠️ **SMOTE** - Feature-based 모델만

**사용 금지**:
7. ❌ **Noise Injection** - 위험
8. ❌ **Landmark Dropout** - 정보 손실
9. ❌ **GAN** - 데이터 부족

---

### 9.2 실행 계획

**Phase 1: Conservative Augmentation (1주)**
```
1. Horizontal Flip
2. Time Stretching (±15%)
→ 10 → 100 samples (10배)
→ Binary classification 재시도
```

**Phase 2: Moderate Augmentation (필요 시)**
```
3. Rotation (±5°)
4. Translation (±5%)
→ 100 → 200 samples (20배)
→ 3-class classification 시도
```

**Phase 3: Evaluation**
```
5. Binary classification accuracy > 70%?
   - Yes → 성공, production
   - No → 추가 데이터 수집 필요
```

---

## 10. 구현 스크립트

**위치**: `scripts/augmentation/augment_leg_agility.py` (생성 예정)

**사용법**:
```bash
python scripts/augmentation/augment_leg_agility.py \
  --input data/leg_agility_train.pkl \
  --output data/leg_agility_train_augmented.pkl \
  --target-count 100 \
  --methods flip,time_stretch,rotation \
  --time-stretch-range 0.85,1.15 \
  --rotation-range -5,5
```

---

**문서 종료**

**다음 단계**: 증강 스크립트 구현 및 실행
