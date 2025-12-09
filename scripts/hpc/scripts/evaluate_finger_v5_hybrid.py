"""
Finger Tapping v5 Hybrid - v3 features + v4 breakpoint features
Combines best of both approaches for better accuracy
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr

# Load v3 data (98 features)
with open('./data/finger_train_v3.pkl', 'rb') as f:
    train_v3 = pickle.load(f)
with open('./data/finger_valid_v3.pkl', 'rb') as f:
    valid_v3 = pickle.load(f)
with open('./data/finger_test_v3.pkl', 'rb') as f:
    test_v3 = pickle.load(f)

# Load v4 data (35 features)
with open('./data/finger_train_v4.pkl', 'rb') as f:
    train_v4 = pickle.load(f)
with open('./data/finger_valid_v4.pkl', 'rb') as f:
    valid_v4 = pickle.load(f)
with open('./data/finger_test_v4.pkl', 'rb') as f:
    test_v4 = pickle.load(f)

print("=" * 70)
print("Finger Tapping v5 Hybrid - v3 + v4 Breakpoint Features")
print("=" * 70)

# Convert to numpy arrays
train_v3['X'] = np.array(train_v3['X'])
train_v3['y'] = np.array(train_v3['y'])
valid_v3['X'] = np.array(valid_v3['X'])
valid_v3['y'] = np.array(valid_v3['y'])
test_v3['X'] = np.array(test_v3['X'])
test_v3['y'] = np.array(test_v3['y'])

train_v4['X'] = np.array(train_v4['X'])
train_v4['y'] = np.array(train_v4['y'])
valid_v4['X'] = np.array(valid_v4['X'])
valid_v4['y'] = np.array(valid_v4['y'])
test_v4['X'] = np.array(test_v4['X'])
test_v4['y'] = np.array(test_v4['y'])

# Check dimensions
print(f"\nv3 features: {train_v3['X'].shape}")
print(f"v4 features: {train_v4['X'].shape}")

# v4 has unique features: breakpoint_norm, slope1, slope2, exp_decay_rate, exp_r2, opening_cv, closing_cv
# These are indices in v4: 4, 5, 6 (breakpoint), 32, 33 (exp), 10, 13 (cv)
# But v3 already has some similar ones

# We'll add only the truly new v4 features to v3:
# 1. breakpoint_norm (idx 4 in v4)
# 2. slope1 (idx 5)
# 3. slope2 (idx 6)
# 4. exp_decay_rate (idx 32)
# 5. exp_r2 (idx 33)
# 6. opening_cv (idx 10 - intra-tap variability)
# 7. closing_cv (idx 13)
# 8. amp_ratio (idx 29)
# 9. sequence_trend (idx 34)
# 10. rhythm_regularity (idx 35 if exists)

# v4 feature names from prepare_finger_v4.py:
v4_features = [
    # Amplitude (7)
    'amp_mean', 'amp_std', 'amp_cv', 'amp_decay_linear', 'breakpoint_norm', 'slope1', 'slope2',
    # Speed (7)
    'opening_mean', 'closing_mean', 'speed_cv', 'opening_cv', 'max_speed', 'min_speed', 'closing_cv',
    # Frequency (7)
    'tap_frequency', 'cycle_mean', 'cycle_std', 'cycle_cv', 'num_taps', 'duration', 'taps_per_sec',
    # Hesitation (7)
    'hesitation_count', 'hesitation_ratio', 'longest_pause', 'mean_pause', 'intertap_cv', 'freeze_count', 'movement_time_ratio',
    # Temporal (7)
    'first5_amp', 'last5_amp', 'amp_ratio', 'exp_decay_rate', 'exp_r2', 'sequence_trend', 'rhythm_regularity'
]

# New features from v4 not in v3 (indices in v4 array)
new_feature_indices = [4, 5, 6, 10, 13, 29, 30, 31, 32, 33, 34]  # breakpoint_norm, slope1, slope2, opening_cv, closing_cv, amp_ratio, exp_decay_rate, exp_r2, sequence_trend, rhythm_regularity

new_feature_names = [v4_features[i] for i in new_feature_indices if i < len(v4_features)]
print(f"\nAdding new v4 features: {new_feature_names}")

# Hybrid features: v3 + new v4 features
def create_hybrid_features(v3_data, v4_data, new_indices):
    v3_features = v3_data['X']
    v4_features = v4_data['X']

    # Extract new features from v4
    new_v4 = v4_features[:, new_indices]

    # Concatenate
    hybrid = np.hstack([v3_features, new_v4])

    # Extract subject from video_id
    subjects = []
    for vid in v3_data['ids']:
        parts = vid.rsplit('_', 1)
        subjects.append(parts[-1] if len(parts) > 1 else vid)

    return {
        'X': hybrid,
        'y': v3_data['y'],
        'subjects': np.array(subjects),
        'ids': v3_data['ids']
    }

train_hybrid = create_hybrid_features(train_v3, train_v4, new_feature_indices)
valid_hybrid = create_hybrid_features(valid_v3, valid_v4, new_feature_indices)
test_hybrid = create_hybrid_features(test_v3, test_v4, new_feature_indices)

print(f"\nHybrid features: {train_hybrid['X'].shape}")
print(f"Total features: 98 (v3) + {len(new_feature_indices)} (new v4) = {train_hybrid['X'].shape[1]}")

# Combine all data for cross-validation
X = np.vstack([train_hybrid['X'], valid_hybrid['X'], test_hybrid['X']])
y = np.hstack([train_hybrid['y'], valid_hybrid['y'], test_hybrid['y']])
subjects = np.hstack([train_hybrid['subjects'], valid_hybrid['subjects'], test_hybrid['subjects']])

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nTotal samples: {len(y)}")
print(f"Total subjects: {len(np.unique(subjects))}")

# Label distribution
print("\nLabel distribution:")
for i in range(5):
    count = np.sum(y == i)
    print(f"  UPDRS {i}: {count} ({100*count/len(y):.1f}%)")

# Models
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

# GroupKFold CV
gkf = GroupKFold(n_splits=5)
results = {}

for name, model in models.items():
    fold_acc = []
    fold_mae = []
    all_preds = []
    all_true = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subjects)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)

        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_mae.append(mean_absolute_error(y_test, y_pred))
        all_preds.extend(y_pred)
        all_true.extend(y_test)

    r, _ = pearsonr(all_true, all_preds)
    results[name] = {
        'accuracy': np.mean(fold_acc),
        'mae': np.mean(fold_mae),
        'pearson': r
    }
    print(f"\n{name}: {len(np.unique(subjects))} subjects, {len(y)} samples")
    print(f"  Accuracy: {100*np.mean(fold_acc):.1f}%")
    print(f"  MAE: {np.mean(fold_mae):.3f}")
    print(f"  Pearson r: {r:.3f}")

# Save hybrid data
os.makedirs('./data', exist_ok=True)
with open('./data/finger_train_v5_hybrid.pkl', 'wb') as f:
    pickle.dump(train_hybrid, f)
with open('./data/finger_valid_v5_hybrid.pkl', 'wb') as f:
    pickle.dump(valid_hybrid, f)
with open('./data/finger_test_v5_hybrid.pkl', 'wb') as f:
    pickle.dump(test_hybrid, f)

print("\n" + "=" * 70)
print("SUMMARY - v5 Hybrid Features")
print("=" * 70)
print(f"{'Model':<20} {'Accuracy':<12} {'MAE':<10} {'Pearson r':<10}")
print("-" * 52)
for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"{name:<20} {100*r['accuracy']:.1f}%{'':<7} {r['mae']:.3f}{'':<5} {r['pearson']:.3f}")

print(f"\nBest v5: {100*max(r['accuracy'] for r in results.values()):.1f}%")
print(f"v3 baseline: 59.2%")
print(f"v4 baseline: 57.8%")
print(f"Improvement over v3: {100*(max(r['accuracy'] for r in results.values()) - 0.592):.1f}%")
