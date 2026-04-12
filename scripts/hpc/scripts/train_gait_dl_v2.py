#!/usr/bin/env python3
"""
Gait Deep Learning v2 - With Data Augmentation and Better Architecture
"""
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("Gait Deep Learning v2 - With Augmentation", flush=True)
print("=" * 70, flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
    test_data = pickle.load(f)

features = train_data['features']
clinical_idx = list(range(99, 129))

X_3d = np.concatenate([
    train_data['X'][:, :, clinical_idx],
    valid_data['X'][:, :, clinical_idx],
    test_data['X'][:, :, clinical_idx]
], axis=0)
y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

n_samples, seq_len, n_features = X_3d.shape
print(f"Data: {X_3d.shape}", flush=True)

# Normalize
X_flat = X_3d.reshape(-1, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, seq_len, n_features)

# Data Augmentation
def augment_data(X, y, num_augment=2):
    """Augment data with noise and time warping"""
    X_aug = [X]
    y_aug = [y]

    for _ in range(num_augment):
        # Add Gaussian noise
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

        # Time shifting (circular shift)
        shift = np.random.randint(-10, 10)
        X_shifted = np.roll(X, shift, axis=1)
        X_aug.append(X_shifted)
        y_aug.append(y)

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

# Improved CNN with Residual connections
class ResidualCNN(nn.Module):
    def __init__(self, input_size, seq_len, num_classes):
        super(ResidualCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = torch.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        identity = x
        x = self.res_block1(x)
        if identity.shape == x.shape:
            x = x + identity
        x = torch.relu(x)
        x = nn.functional.max_pool1d(x, 2)

        x = self.res_block2(x)
        x = torch.relu(x)
        x = nn.functional.max_pool1d(x, 2)

        x = self.res_block3(x)
        x = torch.relu(x)

        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

# Transformer-based model
class GaitTransformer(nn.Module):
    def __init__(self, input_size, seq_len, num_classes, d_model=64, nhead=4, num_layers=2):
        super(GaitTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# LSTM + CNN Hybrid
class HybridModel(nn.Module):
    def __init__(self, input_size, seq_len, num_classes):
        super(HybridModel, self).__init__()

        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # LSTM branch
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, bidirectional=True)

        # Combined
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN
        cnn_out = self.cnn(x.permute(0, 2, 1)).squeeze(-1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Last timestep

        # Combine
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        return self.fc(combined)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(val_true, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return best_val_acc

# Cross-validation with augmentation
print("\n5-Fold CV with Augmentation", flush=True)
print("=" * 70, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(np.unique(y))

models_config = {
    'ResidualCNN': lambda: ResidualCNN(n_features, seq_len, n_classes),
    'Transformer': lambda: GaitTransformer(n_features, seq_len, n_classes),
    'Hybrid_CNN_LSTM': lambda: HybridModel(n_features, seq_len, n_classes),
}

results = {}

for model_name, model_fn in models_config.items():
    print(f"\n[{model_name}]", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        # Augment training data
        X_train_aug, y_train_aug = augment_data(X_scaled[train_idx], y[train_idx], num_augment=2)

        X_train_t = torch.FloatTensor(X_train_aug)
        y_train_t = torch.LongTensor(y_train_aug)
        X_val_t = torch.FloatTensor(X_scaled[val_idx])
        y_val_t = torch.LongTensor(y[val_idx])

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

        model = model_fn().to(device)
        train_model(model, train_loader, val_loader, epochs=100, patience=20)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        acc = accuracy_score(y[val_idx], preds)
        mae = mean_absolute_error(y[val_idx], preds)
        within1 = np.mean(np.abs(y[val_idx] - preds) <= 1)

        fold_scores.append(acc)
        fold_mae.append(mae)
        fold_within1.append(within1)

        print(f"  Fold {fold+1}: Acc={acc:.3f}", flush=True)

    results[model_name] = {
        'accuracy': np.mean(fold_scores),
        'accuracy_std': np.std(fold_scores),
        'mae': np.mean(fold_mae),
        'within1': np.mean(fold_within1)
    }
    print(f"  >> Mean: Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Summary
print("\n" + "=" * 70, flush=True)
print("DEEP LEARNING v2 RESULTS", flush=True)
print("=" * 70, flush=True)

baseline_ml = 0.739

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    diff = acc - baseline_ml
    print(f"{name:<18} {acc:.1%} (vs RF_300: {diff:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest DL v2: {best[0]} with {best[1]['accuracy']:.1%}", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_dl_v2_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved to: {results_path}", flush=True)
