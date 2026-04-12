"""
Train ML models with v4 tapping features for Leg Agility task
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb

# Paths
TRAIN_PATH = "data/leg_agility_train_tapping_v4.pkl"
TEST_PATH = "data/leg_agility_test_tapping_v4.pkl"

def load_data():
    """Load v4 tapping features"""
    print("Loading v4 tapping features...")

    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)

    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {train_data.get('feature_names', 'N/A')}")

    # Feature correlation with score
    print("\nFeature correlations with UPDRS score:")
    feature_names = train_data.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
    for i, name in enumerate(feature_names):
        corr, _ = pearsonr(X_train[:, i], y_train)
        print(f"  {name}: {corr:+.3f}")

    return X_train, y_train, X_test, y_test, feature_names

def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate regression predictions"""
    # Round predictions to nearest integer (0-4)
    y_pred_rounded = np.clip(np.round(y_pred), 0, 4).astype(int)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    exact = accuracy_score(y_true, y_pred_rounded)
    within_1 = np.mean(np.abs(y_true - y_pred_rounded) <= 1)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"MAE:       {mae:.3f}")
    print(f"Exact:     {exact*100:.1f}%")
    print(f"Within 1:  {within_1*100:.1f}%")
    print(f"Pearson:   {pearson_r:.3f}")
    print(f"Spearman:  {spearman_r:.3f}")

    return {
        'mae': mae,
        'exact': exact,
        'within_1': within_1,
        'pearson': pearson_r,
        'spearman': spearman_r
    }

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train Random Forest Regressor"""
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,  # Reduced to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Predict
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("\n--- Train Set ---")
    train_results = evaluate_predictions(y_train, y_pred_train, "Random Forest (Train)")

    print("\n--- Test Set ---")
    test_results = evaluate_predictions(y_test, y_pred_test, "Random Forest (Test)")

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\nFeature Importance:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return rf, test_results

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost Regressor"""
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,  # Reduced to prevent overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    # Predict
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    print("\n--- Train Set ---")
    train_results = evaluate_predictions(y_train, y_pred_train, "XGBoost (Train)")

    print("\n--- Test Set ---")
    test_results = evaluate_predictions(y_test, y_pred_test, "XGBoost (Test)")

    # Feature importance
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\nFeature Importance:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return xgb_model, test_results

def main():
    print("="*60)
    print("Leg Agility v4 Tapping Features - ML Training")
    print("="*60)

    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_data()

    # Train Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test, feature_names)

    # Train XGBoost
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test, feature_names)

    # Comparison
    print("\n" + "="*60)
    print("Model Comparison (Test Set)")
    print("="*60)
    print(f"{'Metric':<15} {'Random Forest':<20} {'XGBoost':<20}")
    print("-"*60)
    for metric in ['mae', 'exact', 'within_1', 'pearson', 'spearman']:
        rf_val = rf_results[metric]
        xgb_val = xgb_results[metric]

        if metric in ['exact', 'within_1']:
            print(f"{metric.upper():<15} {rf_val*100:>18.1f}%  {xgb_val*100:>18.1f}%")
        else:
            print(f"{metric.upper():<15} {rf_val:>20.3f}  {xgb_val:>20.3f}")

    print("\n" + "="*60)
    print("Comparison with Baselines")
    print("="*60)
    print("v2 Mamba + CORAL Pearson: 0.307 (current best)")
    print("v3 Angle Features Pearson: 0.033 (failed)")
    print(f"v4 Random Forest Pearson: {rf_results['pearson']:.3f} ({'+'if rf_results['pearson'] > 0.307 else ''}{(rf_results['pearson']-0.307):.3f})")
    print(f"v4 XGBoost Pearson: {xgb_results['pearson']:.3f} ({'+'if xgb_results['pearson'] > 0.307 else ''}{(xgb_results['pearson']-0.307):.3f})")

    # Determine best
    best_v4 = max(rf_results['pearson'], xgb_results['pearson'])
    if best_v4 > 0.307:
        print(f"\n[OK] v4 IMPROVED over v2! ({best_v4:.3f} > 0.307)")
    else:
        print(f"\n[X] v4 did not improve over v2 ({best_v4:.3f} < 0.307)")

if __name__ == "__main__":
    main()
