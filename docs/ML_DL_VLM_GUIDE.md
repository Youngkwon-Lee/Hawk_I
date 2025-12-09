# ML/DL/VLM Training Guide

Hawkeye 프로젝트의 머신러닝, 딥러닝, VLM 학습 가이드입니다.

## 📊 환경별 학습 가이드

### 학습 환경 매트릭스

| 모델 유형 | 로컬 (CPU/GPU) | HPC (V100/A100) | Cloud (API) |
|----------|---------------|-----------------|-------------|
| **ML** (RF, XGBoost) | ✅ 권장 | ✅ 가능 | - |
| **DL** (LSTM, Transformer) | ⚠️ 테스트만 | ✅ 권장 | ✅ 가능 |
| **VLM** (Qwen-VL, LLaVA) | ❌ 불가 | ✅ 권장 | ✅ API만 |

---

## 🖥️ 로컬 환경 (노트북/데스크톱)

### 적합한 작업
- ML 모델 학습 (Random Forest, XGBoost, Ordinal Regression)
- 데이터 전처리 및 피처 추출
- DL 모델 디버깅 (소규모 데이터)
- VLM API 호출 (GPT-4V, Claude)

### 설치
```bash
# 기본 ML 환경
pip install -r requirements-base.txt

# DL 테스트용 (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 학습 실행
```bash
# ML 모델 학습
python scripts/training/train_finger_tapping_ml.py
python scripts/training/train_gait_ml.py

# DL 모델 테스트 (소규모)
python scripts/training/train_gait_lstm.py --epochs 5 --batch_size 16
```

---

## 🖧 HPC 환경 (GPU 클러스터)

### 적합한 작업
- DL 모델 학습 (LSTM, Transformer, CNN-LSTM)
- VLM 로컬 모델 추론 (Qwen-VL, LLaVA)
- 대규모 Cross-Validation
- Hyperparameter Search

### 워크플로우

#### Step 1: 로컬에서 데이터 준비
```bash
# MediaPipe로 피처 추출 (로컬에서만 가능)
python scripts/hpc/scripts/prepare_data.py
python scripts/hpc/scripts/prepare_gait_data.py

# 결과: scripts/hpc/data/*.pkl
```

#### Step 2: HPC로 전송
```bash
# 전체 hpc 폴더 전송
scp -r scripts/hpc username@hpc:~/hawkeye/

# 또는 필요한 파일만
scp scripts/hpc/data/*.pkl username@hpc:~/hawkeye/data/
scp scripts/hpc/scripts/*.py username@hpc:~/hawkeye/scripts/
```

#### Step 3: HPC 환경 설정
```bash
ssh username@hpc

# Conda 환경 생성 (최초 1회)
conda create -n hawkeye python=3.10
conda activate hawkeye

# PyTorch + CUDA 설치
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 추가 의존성
pip install -r requirements-hpc.txt
```

#### Step 4: 학습 실행
```bash
# 환경변수 설정
export HAWKEYE_ENV=hpc

# GPU 확인
nvidia-smi

# DL 학습
nohup python scripts/train_gait_lstm.py > train.log 2>&1 &

# VLM 평가
python scripts/vlm/evaluate_vlm.py --config experiments/configs/vlm/qwen_vl_evaluation.yaml
```

#### Step 5: 결과 다운로드
```bash
# 로컬에서 실행
scp username@hpc:~/hawkeye/models/*.pth models/trained/
scp username@hpc:~/hawkeye/results/*.csv experiments/results/
```

### HPC Job Script 예시
```bash
#!/bin/bash
#SBATCH --job-name=hawkeye_lstm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load cuda/11.8
conda activate hawkeye

export HAWKEYE_ENV=hpc
python scripts/hpc/scripts/train_gait_lstm.py
```

---

## 🤖 VLM 학습/평가

### 지원 모델

| 모델 | 파라미터 | VRAM 요구 | 환경 |
|------|---------|----------|------|
| Qwen2-VL-7B | 7B | 16GB (4bit) | HPC |
| LLaVA-1.5-13B | 13B | 24GB (4bit) | HPC |
| GPT-4V | - | API | 로컬/Cloud |
| Claude 3 | - | API | 로컬/Cloud |

### 로컬: API 기반 평가
```bash
# OpenAI API 키 설정
export OPENAI_API_KEY=your_key

# GPT-4V 평가
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/qwen_vl_evaluation.yaml \
    --api
```

### HPC: 로컬 모델 평가
```bash
# HPC에서 실행
export HAWKEYE_ENV=hpc

# Qwen-VL 평가 (4-bit 양자화)
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/qwen_vl_evaluation.yaml
```

---

## 📁 실험 관리

### Config 파일 구조
```
experiments/
├── configs/
│   ├── ml/                    # ML 모델 설정
│   │   └── rf_finger_tapping.yaml
│   ├── dl/                    # DL 모델 설정
│   │   └── lstm_gait.yaml
│   └── vlm/                   # VLM 설정
│       └── qwen_vl_evaluation.yaml
└── results/                   # 실험 결과
    ├── ml/
    ├── dl/
    └── vlm/
```

### Config 사용 예시
```bash
# Config 파일로 학습
python scripts/training/train_ml.py --config experiments/configs/ml/rf_finger_tapping.yaml
```

---

## 📦 Requirements 선택 가이드

```bash
# 로컬 ML 작업
pip install -r requirements-base.txt

# 로컬 DL 테스트
pip install -r requirements-dl.txt

# HPC VLM 작업
pip install -r requirements-vlm.txt

# HPC 전체 (추천)
pip install -r requirements-hpc.txt
```

---

## 🔧 환경 설정

### 환경변수
```bash
# 환경 명시적 지정
export HAWKEYE_ENV=local  # local, hpc, cloud

# 데이터 경로 (HPC)
export SCRATCH=/scratch/username

# API 키
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

### 환경 확인
```bash
python scripts/env_config.py
```

출력 예시:
```
============================================================
Hawkeye Environment Configuration
============================================================
Environment: hpc
Project Root: /home/user/hawkeye
Data Root: /scratch/user/hawkeye_data
Model Root: /home/user/hawkeye/models
------------------------------------------------------------
GPU: cuda (batch_size=64)
Mixed Precision: True
============================================================
```

---

## 📈 성능 벤치마크

### ML 모델 (로컬, ~5분)
| 모델 | Task | Accuracy | MAE |
|------|------|----------|-----|
| Random Forest | Finger Tapping | 68% | 0.42 |
| XGBoost | Gait | 65% | 0.48 |

### DL 모델 (HPC V100, ~20분)
| 모델 | Task | Accuracy | MAE |
|------|------|----------|-----|
| LSTM + Attention | Finger Tapping | 71% | 0.38 |
| LSTM + Attention | Gait | 72% | 0.35 |

### VLM 모델 (HPC A100, ~2시간)
| 모델 | Task | Accuracy | MAE |
|------|------|----------|-----|
| Qwen2-VL-7B | All Tasks | TBD | TBD |
| GPT-4V | All Tasks | TBD | TBD |
