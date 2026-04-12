"""
Train ML models with v3 angle-based features for Leg Agility task
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb

# Paths
TRAIN_PATH = "data/leg_agility_train_angles_v3.pkl"
TEST_PATH = "data/leg_agility_test_angles_v3.pkl"

def load_data():
    """Load v3 angle-based features"""
    print("Loading v3 angle-based features...")

    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)

    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {train_data.get('feature_names', 'N/A')[:5]}...")  # Show first 5

    return X_train, y_train, X_test, y_test

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

    # Score distribution
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(y_pred_rounded, return_counts=True)
    for score, count in zip(unique, counts):
        print(f"  Score {score}: {count} ({count/len(y_pred_rounded)*100:.1f}%)")

    return {
        'mae': mae,
        'exact': exact,
        'within_1': within_1,
        'pearson': pearson_r,
        'spearman': spearman_r
    }

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest Regressor"""
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf.fit(X_train, y_train)

    # Predict
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("\n--- Train Set ---")
    evaluate_predictions(y_train, y_pred_train, "Random Forest (Train)")

    print("\n--- Test Set ---")
    results = evaluate_predictions(y_test, y_pred_test, "Random Forest (Test)")

    # Feature importance
    if hasattr(rf, 'feature_importances_'):
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print(f"\nTop 10 Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

    return rf, results

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost Regressor"""
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )

    xgb_model.fit(X_train, y_train)

    # Predict
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    print("\n--- Train Set ---")
    evaluate_predictions(y_train, y_pred_train, "XGBoost (Train)")

    print("\n--- Test Set ---")
    results = evaluate_predictions(y_test, y_pred_test, "XGBoost (Test)")

    # Feature importance
    if hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print(f"\nTop 10 Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

    return xgb_model, results

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)

    # Train XGBoost
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)

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
    print("Comparison with Baseline (v2 Mamba + CORAL)")
    print("="*60)
    print("v2 Baseline Pearson: 0.307")
    print(f"v3 Random Forest Pearson: {rf_results['pearson']:.3f} ({'+'if rf_results['pearson'] > 0.307 else ''}{(rf_results['pearson']-0.307):.3f})")
    print(f"v3 XGBoost Pearson: {xgb_results['pearson']:.3f} ({'+'if xgb_results['pearson'] > 0.307 else ''}{(xgb_results['pearson']-0.307):.3f})")

    # Save best model
    best_model = xgb_model if xgb_results['pearson'] > rf_results['pearson'] else rf_model
    best_name = 'xgb' if xgb_results['pearson'] > rf_results['pearson'] else 'rf'

    model_path = f"models/trained/{best_name}_leg_agility_angles_v3.pkl"
    os.makedirs("models/trained", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nBest model ({best_name.upper()}) saved to: {model_path}")

if __name__ == "__main__":
    main()
