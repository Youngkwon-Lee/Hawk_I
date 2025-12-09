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
