"""
LSTM Training with Correct Annotations_split
Uses train/valid/test split from Annotations_split (no data leakage)
Includes K-Fold Cross-Validation
"""
import os
import sys
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PD4T_ROOT, PD4T_VIDEOS, TRAINED_MODELS_DIR, CACHE_DIR, BACKEND_DIR, ensure_dirs

sys.path.insert(0, str(BACKEND_DIR))

# Paths - CORRECT: Using Annotations_split
ANNOTATION_DIR = str(PD4T_ROOT / "Annotations_split" / "Finger tapping")  # CORRECT!
VIDEO_DIR = str(PD4T_VIDEOS["finger_tapping"])
OUTPUT_DIR = str(TRAINED_MODELS_DIR)

ensure_dirs()

# Config
SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 21
LANDMARK_DIM = 3


def parse_annotation_txt(txt_path):
    """Parse Annotations_split txt file"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_full = parts[0]  # e.g., "15-002802_r_022"
                frames = int(parts[1])
                score = int(parts[2])

                # Parse: 15-002802_r_022 -> video_id=15-002802, hand=r, subject=022
                parts_split = video_id_full.rsplit('_', 2)
                base_id = parts_split[0]  # 15-002802
                hand = parts_split[1]     # r or l
                subject = parts_split[2]  # 022

                video_path = f"{VIDEO_DIR}/{subject}/{base_id}_{hand}.mp4"

                annotations.append({
                    'video_id': video_id_full,
                    'base_id': base_id,
                    'hand': hand,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': video_path
                })
    return annotations


def extract_landmarks_mediapipe(video_path: str, max_frames: int = 300):
    """Extract hand landmarks using MediaPipe"""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    hands.close()

    return np.array(landmarks_list)


def add_velocity_features(landmarks: np.ndarray) -> np.ndarray:
    """Add velocity features"""
    if len(landmarks) < 2:
        return landmarks

    velocity = np.diff(landmarks, axis=0)
    velocity = np.vstack([np.zeros((1, landmarks.shape[1])), velocity])

    # Key landmark velocities
    thumb_vel = velocity[:, 12:15]
    index_vel = velocity[:, 24:27]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1, keepdims=True)
    index_speed = np.linalg.norm(index_vel, axis=1, keepdims=True)

    # Finger distance
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1, keepdims=True)

    features = np.hstack([landmarks, velocity, thumb_speed, index_speed, finger_distance])
    return features


def extract_all_landmarks(annotations, cache_name: str):
    """Extract landmarks from all videos with caching"""
    cache_path = f"{CACHE_DIR}/{cache_name}_landmarks.pkl"

    if os.path.exists(cache_path):
        print(f"Loading cached landmarks from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_landmarks = []
    all_scores = []
    valid_ids = []

    print(f"Extracting landmarks from {len(annotations)} videos...")

    for ann in tqdm(annotations):
        video_path = ann['video_path']

        if not os.path.exists(video_path):
            continue

        try:
            landmarks = extract_landmarks_mediapipe(video_path)
            if len(landmarks) < 10:
                continue

            features = add_velocity_features(landmarks)
            all_landmarks.append(features)
            all_scores.append(ann['score'])
            valid_ids.append(ann['video_id'])

        except Exception as e:
            print(f"Error {ann['video_id']}: {e}")
            continue

    # Cache results
    print(f"Caching {len(all_landmarks)} samples to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump((all_landmarks, all_scores, valid_ids), f)

    return all_landmarks, all_scores, valid_ids


def extract_clinical_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract clinically relevant time-series features for LSTM"""
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    wrist_pos = landmarks[:, 0:3]

    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    dist_velocity = np.gradient(finger_distance)
    dist_accel = np.gradient(dist_velocity)

    thumb_vel = landmarks[:, 63+12:63+15]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1)

    index_vel = landmarks[:, 63+24:63+27]
    index_speed = np.linalg.norm(index_vel, axis=1)

    combined_speed = thumb_speed + index_speed

    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 0.001
    normalized_distance = finger_distance / hand_size

    features = np.stack([
        finger_distance,
        dist_velocity,
        dist_accel,
        thumb_speed,
        index_speed,
        combined_speed,
        thumb_from_wrist,
        index_from_wrist,
        normalized_distance,
        hand_size,
    ], axis=1)

    return features


def pad_sequence(seq, target_len):
    """Pad or truncate sequence to target length"""
    if len(seq) >= target_len:
        return seq[:target_len]
    else:
        padded = np.zeros((target_len, seq.shape[1]))
        padded[:len(seq)] = seq
        return padded


class AttentionLSTM(nn.Module):
    """LSTM with Attention mechanism"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context).squeeze()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        preds = model(X_t).cpu().numpy()
        preds = np.clip(preds, 0, 4)

    mae = np.mean(np.abs(y - preds))
    preds_rounded = np.round(preds)
    exact = np.mean(preds_rounded == y) * 100
    within1 = np.mean(np.abs(y - preds_rounded) <= 1) * 100

    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds}


def kfold_cv(X, y, n_splits=5, device='cpu'):
    """K-Fold Cross-Validation"""
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation (K={n_splits})")
    print('='*60)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)  # Bin for stratification

    fold_results = []
    all_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Normalize
        mean = X_train_fold.mean(axis=(0, 1), keepdims=True)
        std = X_train_fold.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train_fold - mean) / std
        X_val_norm = (X_val_fold - mean) / std

        # Model
        input_size = X.shape[2]
        model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

        # Training
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_fold)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(80):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            model.eval()
            with torch.no_grad():
                val_X_t = torch.FloatTensor(X_val_norm).to(device)
                val_preds = model(val_X_t)
                val_loss = criterion(val_preds, torch.FloatTensor(y_val_fold).to(device)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

        model.load_state_dict(best_state)

        # Evaluate
        results = evaluate(model, X_val_norm, y_val_fold, device)
        all_preds[val_idx] = results['preds']
        fold_results.append(results)

        print(f"  MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%, Within1: {results['within1']:.1f}%")

    # Summary
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])

    print(f"\n{'='*60}")
    print("K-Fold CV Results")
    print('='*60)
    print(f"Average MAE: {avg_mae:.3f} (+/- {np.std([r['mae'] for r in fold_results]):.3f})")
    print(f"Average Exact: {avg_exact:.1f}%")
    print(f"Average Within1: {avg_within1:.1f}%")

    # Overall metrics
    overall_mae = np.mean(np.abs(y - all_preds))
    overall_exact = np.mean(np.round(all_preds) == y) * 100
    overall_within1 = np.mean(np.abs(y - np.round(all_preds)) <= 1) * 100
    pearson, _ = stats.pearsonr(y, all_preds)

    print(f"\nOverall (all folds combined):")
    print(f"  MAE: {overall_mae:.3f}")
    print(f"  Exact: {overall_exact:.1f}%")
    print(f"  Within1: {overall_within1:.1f}%")
    print(f"  Pearson r: {pearson:.3f}")

    return {
        'fold_results': fold_results,
        'avg_mae': avg_mae,
        'avg_exact': avg_exact,
        'avg_within1': avg_within1,
        'overall_mae': overall_mae,
        'overall_exact': overall_exact,
        'overall_within1': overall_within1,
        'pearson': pearson
    }


def train_final_model(X_train, y_train, X_test, y_test, device='cpu'):
    """Train final model on all training data, evaluate on test"""
    print(f"\n{'='*60}")
    print("Training Final Model")
    print('='*60)

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    norm_params = {'mean': mean.squeeze(), 'std': std.squeeze()}

    # Model
    input_size = X_train.shape[2]
    model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

    # Training with validation split
    val_size = int(len(X_train_norm) * 0.15)
    indices = torch.randperm(len(X_train_norm))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm[train_idx]),
        torch.FloatTensor(y_train[train_idx])
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_val = torch.FloatTensor(X_train_norm[val_idx]).to(device)
    y_val = torch.FloatTensor(y_train[val_idx]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Test evaluation
    test_results = evaluate(model, X_test_norm, y_test, device)

    print(f"\n{'='*60}")
    print("Test Set Results")
    print('='*60)
    print(f"MAE: {test_results['mae']:.3f}")
    print(f"Exact Accuracy: {test_results['exact']:.1f}%")
    print(f"Within 1 Point: {test_results['within1']:.1f}%")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, np.round(test_results['preds']).astype(int), labels=range(5))
    print(f"\nConfusion Matrix:")
    print(cm)

    return model, norm_params, test_results


def main():
    print("="*60)
    print("LSTM Training with Correct Annotations_split")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load annotations from Annotations_split (CORRECT!)
    print(f"\nLoading annotations from: {ANNOTATION_DIR}")
    train_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/train.txt")
    valid_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/valid.txt")
    test_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/test.txt")

    print(f"Train: {len(train_ann)}, Valid: {len(valid_ann)}, Test: {len(test_ann)}")

    # Score distribution
    for name, ann in [('Train', train_ann), ('Valid', valid_ann), ('Test', test_ann)]:
        scores = {}
        for a in ann:
            s = a['score']
            scores[s] = scores.get(s, 0) + 1
        print(f"{name} distribution: {dict(sorted(scores.items()))}")

    # Extract landmarks
    print("\n" + "="*60)
    print("Extracting Landmarks")
    print("="*60)

    train_landmarks, train_scores, train_ids = extract_all_landmarks(train_ann, "train_split")
    valid_landmarks, valid_scores, valid_ids = extract_all_landmarks(valid_ann, "valid_split")
    test_landmarks, test_scores, test_ids = extract_all_landmarks(test_ann, "test_split")

    print(f"\nExtracted - Train: {len(train_landmarks)}, Valid: {len(valid_landmarks)}, Test: {len(test_landmarks)}")

    # Extract clinical features and pad
    print("\nExtracting clinical features...")

    def process_landmarks(landmarks_list, target_len=150):
        processed = []
        for lm in landmarks_list:
            clinical = extract_clinical_features(lm)
            padded = pad_sequence(clinical, target_len)
            processed.append(padded)
        return np.array(processed)

    X_train = process_landmarks(train_landmarks)
    X_valid = process_landmarks(valid_landmarks)
    X_test = process_landmarks(test_landmarks)

    y_train = np.array(train_scores)
    y_valid = np.array(valid_scores)
    y_test = np.array(test_scores)

    print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

    # Combine train + valid for K-Fold CV
    X_trainval = np.vstack([X_train, X_valid])
    y_trainval = np.concatenate([y_train, y_valid])

    print(f"Train+Valid combined: {X_trainval.shape}")

    # K-Fold Cross-Validation
    cv_results = kfold_cv(X_trainval, y_trainval, n_splits=5, device=device)

    # Train final model
    model, norm_params, test_results = train_final_model(X_trainval, y_trainval, X_test, y_test, device)

    # Save model
    print(f"\n{'='*60}")
    print("Saving Model")
    print('='*60)

    save_path = f"{OUTPUT_DIR}/lstm_finger_tapping_correct.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[2],
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'norm_params': norm_params,
        'cv_results': {
            'mae': cv_results['avg_mae'],
            'exact': cv_results['avg_exact'],
            'within1': cv_results['avg_within1'],
            'pearson': cv_results['pearson']
        },
        'test_results': {
            'mae': test_results['mae'],
            'exact': test_results['exact'],
            'within1': test_results['within1']
        }
    }, save_path)

    print(f"Model saved to {save_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"K-Fold CV (5-fold):")
    print(f"  MAE: {cv_results['avg_mae']:.3f}")
    print(f"  Exact: {cv_results['avg_exact']:.1f}%")
    print(f"  Within1: {cv_results['avg_within1']:.1f}%")
    print(f"  Pearson: {cv_results['pearson']:.3f}")
    print(f"\nTest Set:")
    print(f"  MAE: {test_results['mae']:.3f}")
    print(f"  Exact: {test_results['exact']:.1f}%")
    print(f"  Within1: {test_results['within1']:.1f}%")


if __name__ == "__main__":
    main()
