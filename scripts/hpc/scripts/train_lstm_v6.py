"""
LSTM Training for v5 (raw) vs v6 (preprocessed) 3D Sequence Data
Compare the effect of preprocessing on model performance
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(context)
        return logits


class TemporalConvLSTM(nn.Module):
    """Temporal Conv + LSTM hybrid"""
    def __init__(self, input_size, hidden_size=128, num_classes=5, dropout=0.3):
        super().__init__()

        # Temporal convolution for local patterns
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)

        # Conv layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = x.transpose(1, 2)  # (batch, seq, hidden)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        combined = torch.cat([h_forward, h_backward], dim=1)

        # Classification
        logits = self.classifier(combined)
        return logits


def extract_subjects(ids):
    """Extract subject IDs from video IDs"""
    subjects = []
    for vid in ids:
        parts = str(vid).rsplit('_', 1)
        subjects.append(parts[-1] if len(parts) > 1 else str(vid))
    return np.array(subjects)


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    # Within-1 accuracy
    within1 = np.mean(np.abs(np.array(all_labels) - np.array(all_preds)) <= 1)

    # Pearson correlation
    if len(set(all_preds)) > 1 and len(set(all_labels)) > 1:
        r, _ = pearsonr(all_labels, all_preds)
    else:
        r = 0.0

    return total_loss / len(dataloader), acc, mae, within1, r, all_preds, all_labels


def run_experiment(data_version, model_class, model_name, epochs=100, batch_size=32):
    """Run training experiment for a specific data version and model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name} on {data_version}")
    print(f"{'='*70}")

    data_dir = './data'

    # Load data
    with open(f'{data_dir}/finger_train_{data_version}.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'{data_dir}/finger_valid_{data_version}.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open(f'{data_dir}/finger_test_{data_version}.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X_train = np.array(train_data['X'])
    y_train = np.array(train_data['y'])
    X_valid = np.array(valid_data['X'])
    y_valid = np.array(valid_data['y'])
    X_test = np.array(test_data['X'])
    y_test = np.array(test_data['y'])

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid),
        torch.LongTensor(y_valid)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    input_size = X_train.shape[2]
    model = model_class(input_size=input_size, hidden_size=128, num_classes=5).to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 5
    weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Training
    best_valid_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc, valid_mae, valid_w1, valid_r, _, _ = evaluate(model, valid_loader, criterion)

        scheduler.step(valid_loss)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Acc={train_acc:.3f}, Valid Acc={valid_acc:.3f}, W1={valid_w1:.3f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    test_loss, test_acc, test_mae, test_w1, test_r, test_preds, test_labels = evaluate(model, test_loader, criterion)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc*100:.1f}%")
    print(f"  MAE: {test_mae:.3f}")
    print(f"  Within-1: {test_w1*100:.1f}%")
    print(f"  Pearson r: {test_r:.3f}")

    return {
        'data_version': data_version,
        'model': model_name,
        'test_acc': test_acc,
        'test_mae': test_mae,
        'test_within1': test_w1,
        'test_r': test_r,
        'test_preds': test_preds,
        'test_labels': test_labels
    }


def main():
    print("=" * 70)
    print("LSTM Training: v5 (raw) vs v6 (preprocessed)")
    print("=" * 70)

    results = []

    # Experiments
    experiments = [
        ('v5_3d', BiLSTMClassifier, 'BiLSTM-Attention'),
        ('v6_preprocessed', BiLSTMClassifier, 'BiLSTM-Attention'),
        ('v5_3d', TemporalConvLSTM, 'Conv-LSTM'),
        ('v6_preprocessed', TemporalConvLSTM, 'Conv-LSTM'),
    ]

    for data_version, model_class, model_name in experiments:
        result = run_experiment(data_version, model_class, model_name, epochs=100)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Preprocessing Effect on LSTM Performance")
    print("=" * 70)
    print(f"{'Data':<20} {'Model':<20} {'Acc':<10} {'MAE':<10} {'W1':<10} {'r':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['data_version']:<20} {r['model']:<20} {r['test_acc']*100:.1f}%{'':<4} {r['test_mae']:.3f}{'':<5} {r['test_within1']*100:.1f}%{'':<4} {r['test_r']:.3f}")

    # v5 vs v6 comparison
    print("\n" + "=" * 70)
    print("v5 vs v6 Comparison")
    print("=" * 70)

    v5_results = [r for r in results if r['data_version'] == 'v5_3d']
    v6_results = [r for r in results if r['data_version'] == 'v6_preprocessed']

    v5_best_acc = max(r['test_acc'] for r in v5_results)
    v6_best_acc = max(r['test_acc'] for r in v6_results)

    v5_best_w1 = max(r['test_within1'] for r in v5_results)
    v6_best_w1 = max(r['test_within1'] for r in v6_results)

    print(f"v5 (raw) Best Accuracy: {v5_best_acc*100:.1f}%")
    print(f"v6 (preprocessed) Best Accuracy: {v6_best_acc*100:.1f}%")
    print(f"Improvement: {(v6_best_acc - v5_best_acc)*100:+.1f}%")
    print()
    print(f"v5 (raw) Best Within-1: {v5_best_w1*100:.1f}%")
    print(f"v6 (preprocessed) Best Within-1: {v6_best_w1*100:.1f}%")
    print(f"Improvement: {(v6_best_w1 - v5_best_w1)*100:+.1f}%")


if __name__ == "__main__":
    main()
