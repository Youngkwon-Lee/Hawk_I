"""
Calculate Spearman Correlation for existing experiments
Compare with Pearson to get accurate PD4T benchmark comparison

Usage:
    python scripts/calculate_spearman.py
"""
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tqdm import tqdm


# ============================================================
# Feature Engineering (same as training scripts)
# ============================================================
class FeatureEngineer:
    @staticmethod
    def compute_enhanced_features(X):
        features_list = []
        for sample in X:
            original = sample
            velocity = np.gradient(sample, axis=0)
            acceleration = np.gradient(velocity, axis=0)

            window = min(10, sample.shape[0] // 3)
            if window < 3:
                window = 3

            moving_mean = np.array([
                np.convolve(sample[:, i], np.ones(window)/window, mode='same')
                for i in range(sample.shape[1])
            ]).T
            moving_std = np.array([
                np.array([sample[max(0,j-window):j+1, i].std()
                         for j in range(sample.shape[0])])
                for i in range(sample.shape[1])
            ]).T
            moving_min = np.array([
                np.array([sample[max(0,j-window):j+1, i].min()
                         for j in range(sample.shape[0])])
                for i in range(sample.shape[1])
            ]).T
            moving_max = np.array([
                np.array([sample[max(0,j-window):j+1, i].max()
                         for j in range(sample.shape[0])])
                for i in range(sample.shape[1])
            ]).T

            enhanced = np.concatenate([
                original, velocity, acceleration,
                moving_mean, moving_std, moving_min, moving_max
            ], axis=1)
            features_list.append(enhanced)
        return np.array(features_list)


# ============================================================
# Mamba Model (same as training)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout, state_size=16, expand_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        expanded = hidden_size * expand_factor
        self.in_proj = nn.Linear(hidden_size, expanded * 2)
        self.conv = nn.Conv1d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)
        self.dt_proj = nn.Linear(expanded, expanded)
        self.A_log = nn.Parameter(torch.randn(expanded, state_size))
        self.D = nn.Parameter(torch.ones(expanded))
        self.B_proj = nn.Linear(expanded, state_size)
        self.C_proj = nn.Linear(expanded, state_size)
        self.out_proj = nn.Linear(expanded, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)
        x_conv = x_path.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self._selective_scan(x_conv)
        z = F.silu(z)
        x = x_ssm * z
        x = self.out_proj(x)
        x = self.dropout(x)
        return x + residual

    def _selective_scan(self, x):
        batch, seq, expanded = x.shape
        device = x.device
        dt = F.softplus(self.dt_proj(x))
        weights = torch.zeros(seq, seq, device=device)
        for i in range(seq):
            for j in range(i + 1):
                weights[i, j] = (0.9 ** (i - j))
        weights = weights[:, :min(seq, 100)]
        x_weighted = x[:, :min(seq, 100), :]
        y_temp = torch.matmul(weights[:seq, :x_weighted.size(1)], x_weighted)
        gate = torch.sigmoid(dt)
        y = y_temp * gate + x * self.D
        return y


class Mamba(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(hidden_size, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


def load_data(task='finger'):
    """Load data for specified task"""
    if task == 'finger':
        possible_paths = [
            ("./finger_train_v4.pkl", "./finger_test_v4.pkl"),
            ("./data/finger_train_v4.pkl", "./data/finger_test_v4.pkl"),
        ]
    else:  # gait
        possible_paths = [
            ("./gait_train.pkl", "./gait_test.pkl"),
            ("./data/gait_train_data.pkl", "./data/gait_test_data.pkl"),
        ]

    for train_path, test_path in possible_paths:
        if os.path.exists(train_path):
            print(f"Loading {task} data from: {train_path}")
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)

            if isinstance(train_data, dict):
                return train_data['X'], train_data['y'], test_data['X'], test_data['y']
            else:
                return train_data, None, test_data, None

    raise FileNotFoundError(f"No {task} data files found!")


def calculate_correlations(task='finger', use_enhanced=True):
    """Run 5-fold CV and calculate both Pearson and Spearman"""
    print(f"\n{'='*60}")
    print(f"Calculating Correlations: {task.upper()} {'+ Enhanced' if use_enhanced else '(Baseline)'}")
    print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    X_train, y_train, _, _ = load_data(task)
    print(f"Data shape: {X_train.shape}")

    # Apply feature engineering
    if use_enhanced:
        print("Applying enhanced features...")
        X_train = FeatureEngineer.compute_enhanced_features(X_train)
        print(f"Enhanced shape: {X_train.shape}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_binned = np.clip(y_train, 0, 3).astype(int)

    all_preds = np.zeros(len(y_train))
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binned)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Normalize
        mean = X_tr.mean(axis=(0, 1), keepdims=True)
        std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        X_tr_norm = (X_tr - mean) / std
        X_val_norm = (X_val - mean) / std

        # Create model
        model = Mamba(X_train.shape[2], 256, 4, 0.4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
        criterion = nn.MSELoss()

        # Quick training (50 epochs for speed)
        model.train()
        for epoch in tqdm(range(50), desc=f"Fold {fold+1}"):
            for i in range(0, len(X_tr_norm), 32):
                batch_X = torch.FloatTensor(X_tr_norm[i:i+32]).to(device)
                batch_y = torch.FloatTensor(y_tr[i:i+32]).to(device)

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_val_norm).to(device)).cpu().numpy()
            preds = np.clip(preds, 0, 4)

        all_preds[val_idx] = preds

        # Calculate both correlations for this fold
        pearson, _ = stats.pearsonr(y_val, preds)
        spearman, _ = stats.spearmanr(y_val, preds)
        mae = np.mean(np.abs(y_val - preds))

        fold_results.append({
            'pearson': pearson,
            'spearman': spearman,
            'mae': mae
        })
        print(f"  Pearson: {pearson:.3f}, Spearman: {spearman:.3f}, MAE: {mae:.3f}")

    # Overall results
    overall_pearson, _ = stats.pearsonr(y_train, all_preds)
    overall_spearman, _ = stats.spearmanr(y_train, all_preds)
    overall_mae = np.mean(np.abs(y_train - all_preds))
    exact = np.mean(np.round(all_preds) == y_train) * 100

    avg_pearson = np.mean([r['pearson'] for r in fold_results])
    avg_spearman = np.mean([r['spearman'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"RESULTS: {task.upper()} {'+ Enhanced' if use_enhanced else '(Baseline)'}")
    print('='*60)
    print(f"MAE:      {avg_mae:.3f}")
    print(f"Exact:    {exact:.1f}%")
    print(f"Pearson:  {avg_pearson:.3f} (overall: {overall_pearson:.3f})")
    print(f"Spearman: {avg_spearman:.3f} (overall: {overall_spearman:.3f})")
    print('='*60)

    return {
        'task': task,
        'enhanced': use_enhanced,
        'mae': avg_mae,
        'exact': exact,
        'pearson': avg_pearson,
        'spearman': avg_spearman,
        'overall_pearson': overall_pearson,
        'overall_spearman': overall_spearman
    }


def main():
    print("="*70)
    print("SPEARMAN vs PEARSON CORRELATION COMPARISON")
    print("="*70)

    results = []

    # Finger Tapping + Enhanced
    try:
        r = calculate_correlations('finger', use_enhanced=True)
        results.append(r)
    except Exception as e:
        print(f"Finger Enhanced failed: {e}")

    # Gait + Enhanced
    try:
        r = calculate_correlations('gait', use_enhanced=True)
        results.append(r)
    except Exception as e:
        print(f"Gait Enhanced failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Correlation Comparison")
    print("="*70)
    print(f"{'Task':<25} {'MAE':>8} {'Exact':>10} {'Pearson':>10} {'Spearman':>10}")
    print("-"*70)

    for r in results:
        name = f"{r['task']} {'+ Enhanced' if r['enhanced'] else ''}"
        print(f"{name:<25} {r['mae']:>8.3f} {r['exact']:>9.1f}% {r['pearson']:>10.3f} {r['spearman']:>10.3f}")

    print("\n" + "="*70)
    print("COMPARISON WITH PD4T SOTA (CoRe + PECoP)")
    print("="*70)
    print(f"{'Task':<20} {'CoRe+PECoP (SRC)':>18} {'Hawkeye (Spearman)':>20} {'Diff':>10}")
    print("-"*70)

    pd4t_sota = {'finger': 49.40, 'gait': 82.33}
    for r in results:
        task = r['task']
        if task in pd4t_sota:
            sota = pd4t_sota[task]
            hawkeye = r['spearman'] * 100  # Convert to same scale
            diff = hawkeye - sota
            print(f"{task:<20} {sota:>18.2f} {hawkeye:>20.2f} {diff:>+10.2f}")

    print("="*70)


if __name__ == "__main__":
    main()
