# PD4T LSTM Training - HPC GPU

HPC Innovation Hub에서 GPU 학습을 위한 패키지입니다.

## 디렉토리 구조

```
hpc/
├── README.md           # 이 파일
├── scripts/
│   ├── prepare_data.py # 로컬에서 데이터 준비
│   └── train_lstm_gpu.py # HPC에서 GPU 학습
├── data/               # 준비된 데이터 (pickle)
│   ├── train_data.pkl
│   └── test_data.pkl
├── models/             # 학습된 모델 저장
└── results/            # 학습 결과 저장
```

## 사용 방법

### Step 1: 로컬에서 데이터 준비

```bash
cd C:/Users/YK/tulip/Hawkeye/hpc
python scripts/prepare_data.py
```

이 스크립트는:
- `Annotations_split/Finger tapping/` 사용 (올바른 split)
- MediaPipe로 랜드마크 추출
- Clinical features 추출 및 padding
- `data/train_data.pkl`, `data/test_data.pkl` 생성

### Step 2: HPC로 파일 전송

```bash
# hpc 폴더 전체 전송
scp -r ./hpc username@hpc:~/hawkeye/

# 또는 개별 전송
scp ./hpc/data/*.pkl username@hpc:~/hawkeye/data/
scp ./hpc/scripts/*.py username@hpc:~/hawkeye/scripts/
```

### Step 3: HPC에서 학습 실행

```bash
# SSH 접속
ssh username@hpc

# 환경 활성화
conda activate triage

# GPU 확인
nvidia-smi

# 학습 실행
cd ~/hawkeye
nohup python scripts/train_lstm_gpu.py > train.log 2>&1 &

# 모니터링
tail -f train.log
```

### Step 4: 결과 다운로드

```bash
# 모델 다운로드
scp username@hpc:~/hawkeye/models/*.pth ./models/

# 결과 다운로드
scp username@hpc:~/hawkeye/results/*.txt ./results/
```

## 학습 옵션

```bash
# 모든 모델 학습 (LSTM, Transformer, ConvLSTM)
python scripts/train_lstm_gpu.py --model all

# 특정 모델만
python scripts/train_lstm_gpu.py --model lstm
python scripts/train_lstm_gpu.py --model transformer
python scripts/train_lstm_gpu.py --model convlstm

# 파라미터 조정
python scripts/train_lstm_gpu.py --epochs 200 --batch_size 32

# Mixed Precision 비활성화 (메모리 문제시)
python scripts/train_lstm_gpu.py --no_amp
```

## 모델 아키텍처

| 모델 | 설명 | 파라미터 |
|------|------|---------|
| **AttentionLSTM** | Bidirectional LSTM + Attention | ~500K |
| **TransformerModel** | Transformer Encoder | ~1M |
| **ConvLSTM** | 1D CNN + LSTM 하이브리드 | ~600K |

## 예상 학습 시간

| 환경 | 데이터 준비 | 5-Fold CV (3모델) |
|------|------------|------------------|
| Local CPU | ~30분 | 2-3시간 |
| HPC GPU (V100) | - | 15-20분 |

## 주의사항

1. **데이터 준비는 로컬에서**: MediaPipe가 필요하므로 로컬에서 `prepare_data.py` 실행
2. **올바른 데이터셋 사용**: 반드시 `Annotations_split/` 사용 (데이터 누수 방지)
3. **GPU 메모리**: V100 16GB에서 batch_size=64 권장, OOM시 32로 감소
4. **체크포인트**: 긴 학습시 체크포인트 기능 활용

## 결과 예시

```
==================================================
FINAL SUMMARY
==================================================
Model                CV MAE      CV Exact    Pearson
-------------------------------------------------------
AttentionLSTM        0.380       70.5%       0.71
TransformerModel     0.395       68.2%       0.68
ConvLSTM             0.372       71.8%       0.73
```
