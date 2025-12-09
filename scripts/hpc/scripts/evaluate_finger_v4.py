"""
Evaluate Finger Tapping with v4 Clinical Features

Uses 35 clinically-validated kinematic features (not raw 3D)
Compares multiple ML approaches:
1. Random Forest
2. Gradient Boosting
3. SVM
4. MLP
5. Tiered Binary Classification (from PMC11260436)

GroupKFold CV with subject-level split
"""
import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "./data"
N_SPLITS = 5
RANDOM_STATE = 42


def load_data_v4():
    """Load v4 finger tapping data"""
    train = pickle.load(open(f"{DATA_DIR}/finger_train_v4.pkl", 'rb'))
    valid = pickle.load(open(f"{DATA_DIR}/finger_valid_v4.pkl", 'rb'))
    test = pickle.load(open(f"{DATA_DIR}/finger_test_v4.pkl", 'rb'))

    # Combine all data
    X = np.vstack([train['X'], valid['X'], test['X']])
    y = np.concatenate([train['y'], valid['y'], test['y']])
    ids = list(train['ids']) + list(valid['ids']) + list(test['ids'])

    # Extract subjects from IDs
    subjects = []
    for vid in ids:
        vid = vid.replace('.mp4', '')
        subject = vid.rsplit('_', 1)[-1]
        subjects.append(subject)
    subjects = np.array(subjects)

    return X, y, subjects, train['features']


def evaluate_model(model, X, y, subjects, model_name):
    """Evaluate model with GroupKFold CV"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    unique_subjects = np.unique(subjects)
    print(f"\n{model_name}: {len(unique_subjects)} subjects, {len(X)} samples")

    all_preds = []
    all_true = []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, subjects)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        model_clone = clone_model(model)
        model_clone.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model_clone.predict(X_val_scaled)

        all_preds.extend(y_pred)
        all_true.extend(y_val)

        acc = accuracy_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        fold_results.append({'acc': acc, 'mae': mae})

    # Overall metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)

    # Pearson correlation
    try:
        r, _ = pearsonr(all_true, all_preds)
    except:
        r = 0

    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Pearson r: {r:.3f}")

    return {'accuracy': acc, 'mae': mae, 'pearson_r': r, 'model': model_name}


def clone_model(model):
    """Clone a sklearn model"""
    from sklearn.base import clone
    return clone(model)


def tiered_binary_classification(X, y, subjects):
    """
    Tiered Binary Classification (PMC11260436)
    Stage 1: Normal (0-1) vs Abnormal (2-4)
    Stage 2: Mild (2) vs Severe (3-4)
    Stage 3: Score 3 vs Score 4
    """
    print("\n" + "="*60)
    print("Tiered Binary Classification")
    print("="*60)

    gkf = GroupKFold(n_splits=N_SPLITS)

    all_preds = []
    all_true = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, subjects)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Stage 1: Normal (0-1) vs Abnormal (2-4)
        y_train_s1 = (y_train >= 2).astype(int)
        clf1 = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf1.fit(X_train_scaled, y_train_s1)
        pred_s1 = clf1.predict(X_val_scaled)

        # Stage 2: For predicted abnormal, Mild (2) vs Severe (3-4)
        # Train on abnormal samples only
        abnormal_mask_train = y_train >= 2
        if abnormal_mask_train.sum() > 0:
            X_train_s2 = X_train_scaled[abnormal_mask_train]
            y_train_s2 = (y_train[abnormal_mask_train] >= 3).astype(int)
            clf2 = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
            clf2.fit(X_train_s2, y_train_s2)
        else:
            clf2 = None

        # Stage 3: For predicted severe, Score 3 vs Score 4
        severe_mask_train = y_train >= 3
        if severe_mask_train.sum() > 0:
            X_train_s3 = X_train_scaled[severe_mask_train]
            y_train_s3 = (y_train[severe_mask_train] == 4).astype(int)
            clf3 = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
            clf3.fit(X_train_s3, y_train_s3)
        else:
            clf3 = None

        # Predict
        fold_preds = []
        for i, x in enumerate(X_val_scaled):
            if pred_s1[i] == 0:
                # Normal - predict 0 or 1 based on probability
                prob_abnormal = clf1.predict_proba([x])[0][1]
                if prob_abnormal < 0.25:
                    pred = 0
                else:
                    pred = 1
            else:
                # Abnormal
                if clf2 is not None:
                    pred_s2 = clf2.predict([x])[0]
                    if pred_s2 == 0:
                        pred = 2  # Mild
                    else:
                        # Severe - distinguish 3 vs 4
                        if clf3 is not None:
                            pred_s3 = clf3.predict([x])[0]
                            pred = 4 if pred_s3 == 1 else 3
                        else:
                            pred = 3
                else:
                    pred = 2

            fold_preds.append(pred)

        all_preds.extend(fold_preds)
        all_true.extend(y_val)

    # Metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)

    try:
        r, _ = pearsonr(all_true, all_preds)
    except:
        r = 0

    # Within-1 accuracy
    within_1 = np.mean(np.abs(all_true - all_preds) <= 1) * 100

    print(f"\nTiered Binary Results:")
    print(f"  Exact Accuracy: {acc*100:.1f}%")
    print(f"  Within-1 Accuracy: {within_1:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Pearson r: {r:.3f}")

    return {'accuracy': acc, 'mae': mae, 'pearson_r': r, 'within_1': within_1, 'model': 'TieredBinary'}


def feature_importance_analysis(X, y, feature_names):
    """Analyze feature importance using Random Forest"""
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 15 Features:")
    for i, idx in enumerate(indices[:15]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Feature groups
    groups = {
        'amplitude': list(range(0, 7)),
        'speed': list(range(7, 14)),
        'frequency': list(range(14, 21)),
        'hesitation': list(range(21, 28)),
        'temporal': list(range(28, 35))
    }

    print("\nImportance by Feature Group:")
    for group_name, indices_group in groups.items():
        group_importance = np.sum(importances[indices_group])
        print(f"  {group_name}: {group_importance:.4f}")


def main():
    print("="*70)
    print("Finger Tapping v4 - Clinical Kinematic Features Evaluation")
    print("="*70)

    # Check if data exists
    if not os.path.exists(f"{DATA_DIR}/finger_train_v4.pkl"):
        print("v4 data not found. Running prepare_finger_v4.py first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/prepare_finger_v4.py"])

    # Load data
    X, y, subjects, feature_names = load_data_v4()
    print(f"\nData shape: {X.shape}")
    print(f"Subjects: {len(np.unique(subjects))}")
    print(f"Features: {len(feature_names)}")

    print(f"\nLabel distribution:")
    for i in range(5):
        count = (y == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y)*100:.1f}%)")

    # Models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=RANDOM_STATE
        ),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE
        ),
    }

    # Evaluate each model
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X, y, subjects, name)
        results.append(result)

    # Tiered Binary Classification
    tiered_result = tiered_binary_classification(X, y, subjects)
    results.append(tiered_result)

    # Feature importance
    feature_importance_analysis(X, y, feature_names)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - v4 Clinical Features")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'MAE':<8} {'Pearson r':<10}")
    print("-"*50)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['model']:<20} {r['accuracy']*100:.1f}%{'':<6} {r['mae']:.3f}{'':<3} {r['pearson_r']:.3f}")

    # Compare with v3 baseline (59.2%)
    best_acc = max(r['accuracy'] for r in results)
    print(f"\nBest v4: {best_acc*100:.1f}% vs v3 baseline: 59.2%")
    improvement = (best_acc - 0.592) / 0.592 * 100
    print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
