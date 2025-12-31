# Trained Models - Hawkeye PD Assessment

## Production Models (K-Fold CV, 2024-12-30)

| Task | File | Pearson | MAE | Usage |
|------|------|---------|-----|-------|
| **Gait** | `gait_coral_raw_kfold_best.pth` | 0.790 | 0.351 | 보행 분석 |
| **Finger** | `finger_coral_raw_kfold_best.pth` | 0.553 | 0.428 | 손가락 태핑 |
| **Hand** | `hand_coral_raw_kfold_best.pth` | 0.598 | 0.475 | 손 움직임 |
| **Leg** | `leg_coral_raw_kfold_best.pth` | 0.238 | 0.549 | 다리 민첩성 |

## Model Architecture

```
MambaCoralModel:
- Input: Raw skeleton (no enhanced features)
- Mamba Blocks: 4 layers, hidden=256
- Output: CORAL Ordinal (4 binary classifiers)
- Loss: CORAL Loss (ordinal regression)
```

## Input Specifications

| Task | Shape | Features |
|------|-------|----------|
| Gait | (300, 30) | 10 landmarks × 3 coords |
| Finger | (150, 123) | Hand + Pose landmarks |
| Hand | (150, 63) | 21 hand landmarks × 3 |
| Leg | (150, 18) | 6 leg landmarks × 3 |

## Loading Example

```python
import torch
from your_model import MambaCoralModel

# Load checkpoint
ckpt = torch.load('gait_coral_raw_kfold_best.pth')

# Create model
model = MambaCoralModel(
    input_size=ckpt['config']['input_size'],
    hidden_size=ckpt['config']['hidden_size'],
    num_layers=ckpt['config']['num_layers'],
    num_classes=ckpt['config']['num_classes'],
    dropout=ckpt['config']['dropout']
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    logits = model(skeleton_tensor)  # (B, T, F)
    probs = torch.sigmoid(logits)
    score = (probs > 0.5).sum(dim=1)  # UPDRS 0-4
```

## Training Details

- Method: 5-Fold Stratified CV
- Epochs: 200 (early stopping, patience=30)
- Optimizer: AdamW (lr=0.0005, weight_decay=0.02)
- Data: PD4T dataset (stratified split)
