# CLAUDE.md - Hawkeye Project

## Project Overview

**Hawkeye**: AI-powered Parkinson's Disease motor assessment system using video analysis.

### Tech Stack
- **Backend**: Python, Flask, MediaPipe, PyTorch, XGBoost
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **ML**: LSTM, Random Forest, XGBoost, Ordinal Regression
- **Data**: PD4T, TULIP datasets

## Project Structure

```
Hawkeye/
├── backend/          # Flask API + AI agents
├── frontend/         # Next.js web app
├── data/             # Datasets (Git ignored)
│   ├── raw/PD4T/    # PD4T dataset
│   └── processed/   # Features & cache
├── models/trained/   # Trained models (Git ignored)
├── scripts/
│   ├── training/    # train_*.py scripts
│   ├── evaluation/  # test/eval scripts
│   └── hpc/         # HPC cluster scripts
├── experiments/      # Configs & results
├── notebooks/        # Analysis notebooks
└── docs/            # Documentation
```

## Quick Commands

```bash
# Backend development
cd backend && python app.py

# Frontend development
cd frontend && npm run dev

# Train models
python scripts/training/train_gait_lstm.py
python scripts/training/train_finger_tapping_ml.py

# Evaluate models
python scripts/evaluation/compare_scores_video.py
```

## Key Files

| File | Purpose |
|------|---------|
| `backend/app.py` | Main API server |
| `backend/agents/orchestrator.py` | AI agent coordination |
| `backend/services/mediapipe_processor.py` | Pose estimation |
| `backend/services/gait_cycle_analyzer.py` | Gait analysis |
| `scripts/training/train_lstm_*.py` | LSTM training |
| `models/trained/*.pkl` | Trained model files |

## Data Paths

- **PD4T Dataset**: `data/raw/PD4T/PD4T/PD4T/`
  - Annotations: `Annotations/{task}/train.csv`, `test.csv`
  - Videos: `Videos/{task}/{subject}/`
- **Features**: `data/processed/features/`
- **Cache**: `data/processed/cache/`

### PD4T Annotation 포맷 (중요!)

**Annotation CSV 형식**: `video_id_subject,frame_count,score`

```
예시: 15-005087_l_042,168,3
       └─────┬────┘ └┬┘ └┬┘
         video_id  subject score(UPDRS)
```

**video_id → 파일 경로 매핑**:
- `video_id`: 15-005087_l → 실제 파일명: `15-005087_l.mp4`
- `subject`: 042 → 폴더: `Videos/{task}/042/`
- **전체 경로**: `Videos/{task}/{subject}/{video_id}.mp4`

**Task별 예시**:
```bash
# Finger Tapping
# annotation: 15-005087_l_042,168,3
data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/042/15-005087_l.mp4

# Gait
# annotation: 15-001760_009,547,0
data/raw/PD4T/PD4T/PD4T/Videos/Gait/009/15-001760.mp4
```

**Subject 폴더 목록 (실제 확인)**:
- Finger Tapping: 001-050 (001, 002, 004, 009, 010, ...)
- Gait: 001-050 (유사 구조)

**주의**: annotation의 subject 번호가 폴더명과 직접 매핑됨!

## IMPORTANT: Data Split Policy

**항상 STRATIFIED 데이터 사용** (2024-12-09 결정)

원본 PD4T에 Data Leakage 존재 (환자 중복):
- Finger Tapping: 1명, Gait: 1명, Leg Agility: 5명

### 사용할 파일
```
# Features (학습용)
data/processed/features/*_stratified.csv  ← 사용

# Annotations (참조용)
Annotations/{task}/stratified/            ← 사용
```

### 사용하지 말 것
```
# 원본 (leakage 있음)
*_features.csv (without _stratified)
Annotations/{task}/train.csv, test.csv
```

### 검증 명령어
```bash
python scripts/data_validator.py
python scripts/fix_data_split_stratified.py
```

## Data Pipeline 구조

```
[Video] → [MediaPipe Skeleton] → [Features] → [ML Model]
   ↓              ↓                   ↓
PD4T/Videos/   (실시간 추출)    *_features.csv
```

### 데이터 위치
| 단계 | 위치 | 설명 |
|------|------|------|
| Raw Videos | `data/raw/PD4T/PD4T/Videos/{task}/` | 원본 동영상 |
| Skeleton | (실시간 추출) | MediaPipe로 프레임별 추출 |
| Features | `data/processed/features/` | 집계된 kinematic features |
| Models | `models/trained/*.pkl` | 학습된 ML 모델 |

### Feature 파일 구조
```
data/processed/features/
├── _original_leaky/              # 원본 (사용 금지)
├── finger_tapping_*_features.csv # 기본 (stratified)
├── *_stratified.csv              # stratified 원본
└── *_corrected.csv               # 중간 버전
```

### 주요 Features (MediaPipe 기반)
- `peak_velocity_mean`: 최대 속도 평균
- `opening_velocity_mean`: 손가락 벌림 속도
- `closing_velocity_mean`: 손가락 오므림 속도
- `amplitude_mean/std`: 움직임 진폭
- `rhythm_variability`: 리듬 변동성
- `fatigue_rate`: 피로도 (시간에 따른 감소)

### Feature 추출 스크립트
```bash
# Finger Tapping features
python scripts/extract_finger_tapping_features.py

# All tasks
python scripts/extract_features.py
```

### 기술 문서
- **[Technical Report](docs/TECHNICAL_REPORT.md)**: 종합 기술 보고서 (Dataset, Preprocessing, Models, GPU)
- **[Skeleton & Features 분석](docs/SKELETON_FEATURES_ANALYSIS.md)**: MediaPipe keypoints, 2D/3D 좌표, 정규화 방법, 연구 레퍼런스

### Trained Models (2024-12-09)
| 파일 | 용도 |
|------|------|
| `rf_finger_tapping_scorer.pkl` | Random Forest (Finger Tapping) |
| `xgb_finger_tapping_scorer.pkl` | XGBoost (Finger Tapping) |
| `rf_gait_scorer.pkl` | Random Forest (Gait) |
| `xgb_gait_scorer.pkl` | XGBoost (Gait) |
| `*_scaler.pkl` | Feature Scaler |

## Recent Model Training Results (2025-12-16 ~ 2025-12-23)

### Best Performing Models (HPC Training)

#### Gait Task
| Model | MAE | Exact | Within1 | Pearson | Status |
|-------|-----|-------|---------|---------|--------|
| **CORAL Ordinal** | **0.241** | **76.5%** | **100%** | **0.807** | ✅ **Production** |
| Mamba + Enhanced | 0.335 | 71.9% | 99.4% | 0.804 | ✅ Baseline |
| ActionMamba (Mamba+GCN) | 0.342 | 69.7% | 98.8% | 0.699 | ❌ Unstable |

**Decision**: Use **CORAL Ordinal** for Gait (best overall performance)

#### Finger Tapping Task
| Model | MAE | Exact | Within1 | Pearson | Status |
|-------|-----|-------|---------|---------|--------|
| **Mamba + Enhanced Features** | 0.444 | 63.0% | 97.9% | **0.609** | ✅ **Production** |
| CORAL Ordinal | 0.370 | 64.8% | 98.4% | 0.555 | ✅ Best MAE/Exact |
| ActionMamba (Mamba+GCN) | 0.380 | 64.3% | 97.9% | 0.507 | ❌ Worse |
| Mamba + Clinical V1 | 0.454 | 63.7% | 98.2% | 0.578 | ⚠️ Worse |

**Decision**: Use **Mamba + Enhanced Features** for Finger Tapping (best Pearson 0.609)

#### Hand Movement Task
| Model | MAE | Exact | Within1 | Pearson | Status |
|-------|-----|-------|---------|---------|--------|
| **CORAL Ordinal** | **0.431** | **59.1%** | **97.8%** | **0.593** | ✅ **Production** |
| ActionMamba (Mamba+GCN) | 0.481 | 54.5% | 97.6% | 0.511 | ❌ Worse |

**Decision**: Use **CORAL Ordinal** for Hand Movement (Pearson 0.593)

#### Leg Agility Task
| Model | MAE | Exact | Within1 | Pearson | Status |
|-------|-----|-------|---------|---------|--------|
| CORAL Ordinal | **0.462** | **59.5%** | **96.0%** | **0.221** | ⚠️ **낮은 성능** |
| ActionMamba (Mamba+GCN) | 0.486 | 55.7% | 96.4% | 0.195 | ❌ **완전 실패** |

**Decision**: **Leg Agility Task는 두 모델 모두 실패** (Pearson 0.221/0.195 거의 랜덤)
**Note**: 데이터 자체에 문제 있을 가능성 높음 (작은 샘플, 노이즈, 또는 feature 부족)

### ActionMamba Architecture (Mamba + GCN Hybrid)

**Implementation**: `scripts/train_action_mamba_{task}.py`

**Architecture Components**:
```
Input (B, T, J, 3)
  ↓
ACE (Action Characteristic Encoder)
  ├─ Spatial: GCN (Graph Convolution on skeleton)
  └─ Temporal: MambaBlock (State Space Model)
  ↓
Fusion Layer (Learnable weights α, β)
  ↓
CORAL Ordinal Regression (K-1 binary classifiers)
  ↓
UPDRS Score (0-4)
```

**Key Features**:
- **Spatial GCN**: Normalized adjacency matrix for skeleton topology
- **Temporal Mamba**: Linear complexity O(T) for long sequences
- **CORAL**: Ordinal regression treating 5-class as 4 binary problems
- **Mixed Precision**: FP16 training with GradScaler

**Reference**: Based on "ActionMamba: Hybrid GCN-Mamba for Skeleton Action Recognition" (2025)

**Final Results Summary**:

| Task | ActionMamba Pearson | Best Baseline | Winner | Notes |
|------|-------------------|---------------|--------|-------|
| **Gait** | 0.699 | 0.807 (CORAL) | ❌ Baseline | -13.4% worse |
| **Finger** | 0.507 | 0.609 (Mamba+Enh) | ❌ Baseline | -16.7% worse |
| **Hand** | 0.511 | **0.593 (CORAL)** | ❌ Baseline | -13.8% worse |
| **Leg** | 0.195 | **0.221 (CORAL)** | ❌ Baseline | -11.8% worse (both failed) |

**Conclusion**: **ActionMamba 전면 폐기 (4/4 Task 실패)**

### 실패 원인
1. **GCN + Mamba 조합 비효율**: Spatial GCN이 skeleton topology 활용 못함
2. **CORAL loss 궁합 문제**: Raw skeleton에서만 효과적, complex features와 충돌
3. **Task 특성 불일치**: Action recognition ≠ Medical scoring (UPDRS)
4. **Overfitting**: 복잡한 아키텍처가 일반화 성능 저하

### Lesson Learned
- ❌ **복잡한 아키텍처 ≠ 높은 성능**: 단순한 모델(CORAL, Mamba+Enhanced)이 더 효과적
- ❌ **SOTA 방법론 맹신**: 도메인 특성 무시하면 실패
- ✅ **의료 AI 특수성**: 패턴 인식과 의료 평가는 다른 접근 필요
- ✅ **Baseline 비교 필수**: 구현 전 baseline 결과 확보로 시간 낭비 방지

### 권장 모델 (Production)
- **Gait**: CORAL Ordinal (Pearson 0.807) ✅
- **Finger**: Mamba + Enhanced Features (Pearson 0.609) ✅
- **Hand**: CORAL Ordinal (Pearson 0.593) ✅
- **Leg**: CORAL Ordinal (Pearson 0.221) ⚠️ **사용 주의** (낮은 성능, 데이터 개선 필요)

### Known Issues and Solutions

#### Issue 1: MambaBlock Gradient Explosion (loss=nan)

**Problem**: Training fails with loss=nan at epoch 4-10
**Root Cause**: SSM (State Space Model) state explosion in MambaBlock

**Symptoms**:
```
Epoch 1: loss=0.686, mae=0.676 ✅
Epoch 2: loss=0.557, mae=0.592 ✅
Epoch 3: loss=0.422, mae=0.676 ✅
Epoch 4: loss=nan, mae=0.676 ❌  <- Gradient explosion
```

**Solution** (4-layer fix applied to all ActionMamba scripts):

1. **Exponential Clamping** (Line ~290):
```python
decay = torch.exp(torch.clamp(-dt[:, t].mean(dim=-1, keepdim=True), min=-10, max=10))
```

2. **State Clamping** (Line ~293):
```python
state = torch.clamp(state, min=-10, max=10)  # Prevent state explosion
```

3. **Gradient Clipping** (Line ~472):
```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

4. **Learning Rate Reduction**:
```python
LEARNING_RATE = 0.0001  # Reduced from 0.0005
```

**Result**: Loss=nan completely eliminated, training stable for 200 epochs

**Git Commits**:
- `76e8807`: Initial fix (exponential clamping + gradient clipping + LR reduction)
- `2509406`: Final fix (state clamping added)

#### Issue 2: Script Truncation Bug

**Problem**: Hand/Finger scripts immediately exit without executing
**Root Cause**: Scripts truncated at line 670, missing `if __name__ == "__main__"` block

**Symptoms**:
- Log file only shows "nohup: ignoring input" (22 bytes)
- No training output or errors

**Detection**:
```bash
wc -l train_action_mamba_hand.py    # 670 lines (incomplete)
wc -l train_action_mamba_gait.py    # 697 lines (complete)
grep -n "if __name__" train_action_mamba_hand.py  # Not found
```

**Solution**: Added missing 16 lines (671-686) including result saving and `if __name__ == "__main__": main()` call

**Git Commit**: `adb6bce` - "fix: Add missing main() call to Hand/Finger scripts"

#### Issue 3: Finger Tapping Unexpected Joint Count

**Problem**: Expected 21 hand landmarks but loaded 41 joints
**Possible Cause**: Combined MediaPipe Pose (10 keypoints) + Hand (21 landmarks) + extra features
**Status**: Training proceeding normally, will evaluate results after completion

### HPC Training Logs

**Location**: `scripts/hpc/results/`

**Recent Results**:
- `mamba_enhanced_20251216_114405.txt` - Finger Tapping Mamba
- `mamba_gait_enhanced_20251216_201238.txt` - Gait Mamba
- `mamba_clinical_v1_20251216_185746.txt` - Clinical features experiment

**HPC Deployment Scripts**:
- `scripts/hpc/scripts/train_action_mamba_gait_hpc.sh`
- `scripts/hpc/scripts/train_action_mamba_hand_hpc.sh`
- `scripts/hpc/scripts/train_action_mamba_finger_hpc.sh`
- `scripts/hpc/scripts/train_action_mamba_leg_hpc.sh`

**Execution**:
```bash
# On HPC
cd ~/hawkeye
nohup bash scripts/hpc/scripts/train_action_mamba_{task}_hpc.sh > {task}_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Development Notes

### MDS-UPDRS Tasks
1. **Gait** - Walking pattern (0-4 score)
2. **Finger Tapping** - Repetitive movements (0-4)
3. **Hand Movement** - Open/close (0-4)
4. **Leg Agility** - Leg lifting (0-4)

### Model Training Pipeline
1. Extract features: `scripts/extract_features.py`
2. Train model: `scripts/training/train_*.py`
3. Evaluate: `scripts/evaluation/compare_scores_video.py`
4. Deploy to: `models/trained/`

## Environment

```bash
# Backend
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

## HPC Deployment Notes ⚠️

### Git on HPC
**IMPORTANT**: `git pull origin main` **does NOT work on HPC**
- Reason: SSH port 22 blocked, git clone not initialized
- **Solution**: Use `wget` to download individual files from GitHub raw URLs

**HPC Deployment Workflow**:
```bash
# On Local
git add . && git commit -m "message" && git push origin main

# On HPC - Use wget for individual files
cd ~/hawkeye/scripts
wget -O train_hand_ordinal.py "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/train_hand_ordinal.py"

# Or download multiple files
for file in train_hand_ordinal.py train_leg_ordinal.py; do
  wget -O $file "https://raw.githubusercontent.com/Youngkwon-Lee/Hawk_I/main/scripts/$file"
done
```

### Data Files on HPC
- **Already uploaded**: `hand_movement_train.pkl`, `hand_movement_test.pkl`, `leg_agility_train.pkl`, `leg_agility_test.pkl`
- **Location**: `~/hawkeye/` and `~/hawkeye/data/`
- **No need to re-upload** unless data changes

## Custom Commands (Claude Code)

`.claude/commands/` 폴더의 워크플로우 명령어:

| 명령어 | 설명 |
|--------|------|
| `/train-ml` | ML 모델 학습 (로컬) |
| `/train-dl` | DL 모델 학습 (HPC) |
| `/eval-vlm` | VLM 평가 워크플로우 |

## Environment Configuration

```bash
# 환경 확인
python scripts/env_config.py

# 환경 설정 (HPC)
export HAWKEYE_ENV=hpc

# 환경 설정 (로컬)
export HAWKEYE_ENV=local
```

### Training Environment Matrix

| 모델 유형 | 로컬 | HPC | Cloud API |
|----------|------|-----|-----------|
| ML (RF, XGBoost) | ✅ 권장 | ✅ | - |
| DL (LSTM, Transformer) | ⚠️ 테스트만 | ✅ 권장 | - |
| VLM (Qwen, LLaVA) | ❌ | ✅ 권장 | ✅ API |

## Git Workflow

```bash
# Large files are ignored (data/, models/)
# Only code and configs are tracked
git add .
git commit -m "feat: description"
git push origin main
```
