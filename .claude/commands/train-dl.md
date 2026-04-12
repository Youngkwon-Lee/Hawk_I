# DL Model Training Command (HPC)

딥러닝 모델 학습 워크플로우 (HPC GPU 환경)

## 로컬 준비 단계
1. 데이터 전처리: `python scripts/hpc/scripts/prepare_data.py`
2. HPC 전송: `scp -r scripts/hpc username@hpc:~/hawkeye/`

## HPC 실행 단계
1. 환경 설정:
   ```bash
   export HAWKEYE_ENV=hpc
   conda activate hawkeye
   ```
2. GPU 확인: `nvidia-smi`
3. 학습 실행:
   ```bash
   # Interactive
   python scripts/train_gait_lstm.py

   # SLURM Job
   sbatch scripts/hpc/jobs/train_lstm.sh
   ```

## 주요 모델
- LSTM + Attention
- Transformer
- CNN-LSTM Hybrid

## 결과 다운로드
```bash
scp username@hpc:~/hawkeye/models/*.pth models/trained/
scp username@hpc:~/hawkeye/results/*.csv experiments/results/dl/
```

## 예상 시간
- LSTM: ~20분 (V100 GPU)
- Transformer: ~40분 (V100 GPU)
