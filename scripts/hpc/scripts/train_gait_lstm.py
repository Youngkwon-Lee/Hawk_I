#!/usr/bin/env python3
"""
Gait LSTM Training - Deep Learning on 3D sequence data
Uses clinical features only (30 features × 300 timesteps)
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
print("Gait LSTM Training - Clinical Features (3D Sequence)", flush=True)
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
print("=" * 70, flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Load 3D data (samples × seq_len × features)
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
    test_data = pickle.load(f)

features = train_data['features']
clinical_idx = list(range(99, 129))  # 30 clinical features

# Extract clinical features only
X_train_3d = train_data['X'][:, :, clinical_idx]  # (n, seq, 30)
X_valid_3d = valid_data['X'][:, :, clinical_idx]
X_test_3d = test_data['X'][:, :, clinical_idx]

X_3d = np.concatenate([X_train_3d, X_valid_3d, X_test_3d], axis=0)
y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

print(f"\nData shape: {X_3d.shape} (samples × seq_len × features)", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)

# Normalize per feature across all samples and timesteps
n_samples, seq_len, n_features = X_3d.shape
X_flat = X_3d.reshape(-1, n_features)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)
X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)

print(f"Normalized shape: {X_scaled.shape}", flush=True)

# LSTM Model
class GaitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(GaitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        out = self.fc(context)
        return out

# Simple LSTM (without attention)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# 1D CNN Model
class GaitCNN(nn.Module):
    def __init__(self, input_size, seq_len, num_classes):
        super(GaitCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x).squeeze(-1)
        return self.fc(x)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return best_val_acc

# Cross-validation
print("\n" + "=" * 70, flush=True)
print("5-Fold Stratified Cross-Validation", flush=True)
print("=" * 70, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(np.unique(y))

models_config = {
    'SimpleLSTM': lambda: SimpleLSTM(n_features, hidden_size=64, num_layers=2, num_classes=n_classes),
    'BiLSTM_Attn': lambda: GaitLSTM(n_features, hidden_size=64, num_layers=2, num_classes=n_classes),
    'CNN_1D': lambda: GaitCNN(n_features, seq_len, num_classes=n_classes),
}

results = {}

for model_name, model_fn in models_config.items():
    print(f"\n[{model_name}]", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train_fold = torch.FloatTensor(X_scaled[train_idx])
        y_train_fold = torch.LongTensor(y[train_idx])
        X_val_fold = torch.FloatTensor(X_scaled[val_idx])
        y_val_fold = torch.LongTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = model_fn().to(device)
        best_acc = train_model(model, train_loader, val_loader, epochs=100, patience=15)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_fold.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        acc = accuracy_score(y[val_idx], preds)
        mae = mean_absolute_error(y[val_idx], preds)
        within1 = np.mean(np.abs(y[val_idx] - preds) <= 1)

        fold_scores.append(acc)
        fold_mae.append(mae)
        fold_within1.append(within1)

        print(f"  Fold {fold+1}: Acc={acc:.3f}, MAE={mae:.3f}, Within-1={within1:.3f}", flush=True)

    results[model_name] = {
        'accuracy': np.mean(fold_scores),
        'accuracy_std': np.std(fold_scores),
        'mae': np.mean(fold_mae),
        'within1': np.mean(fold_within1)
    }
    print(f"  >> Mean: Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Summary
print("\n" + "=" * 70, flush=True)
print("DEEP LEARNING RESULTS", flush=True)
print("=" * 70, flush=True)

baseline_ml = 0.739  # Best ML model (RF_300)

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    diff = acc - baseline_ml
    print(f"{name:<15} {acc:.1%} (vs RF_300: {diff:+.1%}), MAE={res['mae']:.3f}, Within-1={res['within1']:.1%}", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest DL: {best[0]} with {best[1]['accuracy']:.1%}", flush=True)
print(f"Best ML (RF_300): 73.9%", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_deep_learning_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to: {results_path}", flush=True)
