# Model Benchmark Command

ML 모델 성능 벤치마크 워크플로우

## 모델 정확도 비교

```bash
cd backend

# Finger Tapping 모델 비교
python -c "
from services.ml_scorer import get_ml_scorer
import pandas as pd
import numpy as np

scorer = get_ml_scorer()
scorer.load_models()

# Load test data
df = pd.read_csv('../data/processed/features/finger_tapping_test_features.csv')

# Feature columns
feature_cols = [
    'tapping_speed', 'amplitude_mean', 'amplitude_std', 'amplitude_decrement',
    'first_half_amplitude', 'second_half_amplitude', 'opening_velocity_mean',
    'closing_velocity_mean', 'peak_velocity_mean', 'velocity_decrement',
    'rhythm_variability', 'hesitation_count', 'halt_count', 'freeze_episodes',
    'fatigue_rate', 'velocity_first_third', 'velocity_mid_third', 'velocity_last_third',
    'amplitude_first_third', 'amplitude_mid_third', 'amplitude_last_third',
    'velocity_slope', 'amplitude_slope', 'rhythm_slope',
    'variability_first_half', 'variability_second_half', 'variability_change'
]

# Evaluate
y_true = df['score'].values
rf_preds = []
xgb_preds = []

for _, row in df.iterrows():
    metrics = row[feature_cols].to_dict()
    rf_result = scorer.predict_finger_tapping(metrics, 'rf')
    xgb_result = scorer.predict_finger_tapping(metrics, 'xgb')
    rf_preds.append(rf_result.score if rf_result else -1)
    xgb_preds.append(xgb_result.score if xgb_result else -1)

from sklearn.metrics import accuracy_score, mean_absolute_error

print('=== Finger Tapping Model Benchmark ===')
print(f'RF  - Accuracy: {accuracy_score(y_true, rf_preds):.3f}, MAE: {mean_absolute_error(y_true, rf_preds):.3f}')
print(f'XGB - Accuracy: {accuracy_score(y_true, xgb_preds):.3f}, MAE: {mean_absolute_error(y_true, xgb_preds):.3f}')
"
```

## Rule vs ML vs Ensemble 비교

```bash
python -c "
from services.updrs_scorer import UPDRSScorer
from services.metrics_calculator import MetricsCalculator
import pandas as pd
import numpy as np

# Load test data
df = pd.read_csv('../data/processed/features/finger_tapping_test_features.csv')
calc = MetricsCalculator(fps=30.0)

rule_scores = []
ml_scores = []
ensemble_scores = []
y_true = df['score'].values

# This is a simplified version - real benchmark needs actual landmark data
print('Run full benchmark with: python scripts/evaluation/compare_scores_video.py')
"
```

## 결과 위치
- Metrics: `experiments/results/ml/`
- Confusion Matrix: `experiments/results/ml/confusion_matrix.png`
