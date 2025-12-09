#!/usr/bin/env python3
"""
Temporal Attention Model for PD4T
- Lightweight models for small datasets (30 subjects)
- Uses all features (coords + clinical)
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import time
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='both', choices=['gait', 'finger', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--cv', type=str, default='loso', choices=['loso', '5fold'])
    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== MODELS =====================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x)

class TemporalCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x).squeeze(-1)
        return self.classifier(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        T = x.size(1)
        x = self.input_proj(x) + self.pos_encoding[:, :T, :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)

# ===================== DATASET =====================

class PD4TDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]

# ===================== DATA & TRAINING =====================

def load_data(task, data_dir):
    X_all, y_all, ids_all = [], [], []
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{task}_{split}_v2.pkl', 'rb') as f:
            d = pickle.load(f)
        X_all.append(d['X'])
        y_all.append(d['y'])
        ids_all.append(d['ids'])

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    subjects = np.array([id.rsplit('_', 1)[-1] for id in np.hstack(ids_all)])

    # Normalize
    X = (X - X.mean(axis=(0,1), keepdims=True)) / (X.std(axis=(0,1), keepdims=True) + 1e-6)
    return X, y, subjects

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            targets.extend(y.numpy())
    return np.array(preds), np.array(targets)

def run_cv(model_class, model_name, task, args, data_dir):
    X, y, subjects = load_data(task, data_dir)
    input_dim, num_classes = X.shape[2], len(np.unique(y))

    cv = LeaveOneGroupOut() if args.cv == 'loso' else GroupKFold(n_splits=5)
    fold_accs = []
    all_preds, all_targets = [], []

    for train_idx, test_idx in cv.split(X, y, subjects):
        if len(test_idx) == 0:
            continue

        train_loader = DataLoader(PD4TDataset(X[train_idx], y[train_idx]), batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(PD4TDataset(X[test_idx], y[test_idx]), batch_size=len(test_idx))

        model = model_class(input_dim, args.hidden, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        for _ in range(args.epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

        preds, targets = evaluate(model, test_loader, device)
        fold_accs.append(accuracy_score(targets, preds))
        all_preds.extend(preds)
        all_targets.extend(targets)

    return {
        'model': model_name,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': accuracy_score(all_targets, all_preds),
        'within1': np.mean(np.abs(np.array(all_targets) - np.array(all_preds)) <= 1)
    }

def run_task(task, args, data_dir):
    print('=' * 60)
    print(f'{task.upper()} - Temporal Models Comparison')
    print('=' * 60)

    X, y, subjects = load_data(task, data_dir)
    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}, Features: {X.shape[2]}')
    print()

    models = [
        (AttentionLSTM, 'Attention-LSTM'),
        (TemporalCNN, 'Temporal-CNN'),
        (TransformerEncoder, 'Transformer'),
    ]

    results = []
    for model_class, model_name in models:
        print(f'Training {model_name}...', end=' ', flush=True)
        t0 = time.time()
        r = run_cv(model_class, model_name, task, args, data_dir)
        print(f'{r["mean_acc"]:.1%} ± {r["std_acc"]:.1%} ({time.time()-t0:.0f}s)')
        results.append(r)

    best = max(results, key=lambda x: x['mean_acc'])
    print(f'\nBest: {best["model"]} - {best["mean_acc"]:.1%}')
    return results

def main():
    args = parse_args()
    data_dir = Path(__file__).parent.parent / 'data'

    print('=' * 60)
    print('PD4T - Temporal Models')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 60)

    results = {}
    if args.task in ['gait', 'both']:
        results['gait'] = run_task('gait', args, data_dir)
    if args.task in ['finger', 'both']:
        print()
        results['finger'] = run_task('finger', args, data_dir)

    print()
    print('=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    for task in ['gait', 'finger']:
        if task not in results:
            continue
        baseline = 67.1 if task == 'gait' else 59.1
        best_ml = 70.4 if task == 'gait' else 58.3
        print(f'\n{task.upper()} (Baseline: {baseline}%, Best ML: {best_ml}%):')
        for r in results[task]:
            marker = '★' if r['mean_acc'] * 100 > best_ml else ''
            print(f'  {r["model"]}: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%} {marker}')

if __name__ == '__main__':
    main()
