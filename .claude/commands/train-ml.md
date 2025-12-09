# ML Model Training Command

ML 모델 학습 워크플로우 (로컬 환경)

## 실행 단계
1. 환경 확인: `python scripts/env_config.py`
2. 데이터 검증: PD4T 데이터셋 경로 확인
3. 학습 실행:
   - Finger Tapping: `python scripts/training/train_finger_tapping_ml.py`
   - Gait: `python scripts/training/train_gait_ml.py`
4. 결과 확인: `experiments/results/ml/` 폴더

## 주요 모델
- Random Forest
- XGBoost
- Ordinal Regression

## 예상 시간
- 전체 학습: ~5분 (로컬 CPU)
