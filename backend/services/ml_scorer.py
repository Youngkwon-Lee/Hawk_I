"""
ML-Based UPDRS Scorer
Uses trained ML models for UPDRS score prediction
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

ML_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml_models")


@dataclass
class MLPrediction:
    score: int
    confidence: float
    raw_prediction: float
    model_type: str


class MLScorer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLScorer, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._scalers = {}
            cls._instance._feature_cols = {}
            cls._instance._loaded = False
        return cls._instance

    def load_models(self, model_dir: str = ML_MODEL_DIR) -> bool:
        if self._loaded:
            return True
        try:
            import joblib
        except ImportError:
            return False

        ft_models = [("finger_tapping_rf", "rf_finger_tapping_scorer.pkl"),
                     ("finger_tapping_xgb", "xgb_finger_tapping_scorer.pkl"),
                     ("finger_tapping_ordinal", "ordinal_finger_tapping_scorer.pkl")]
        for key, fn in ft_models:
            path = os.path.join(model_dir, fn)
            if os.path.exists(path):
                try:
                    self._models[key] = joblib.load(path)
                except:
                    pass

        for name, fn in [("finger_tapping", "finger_tapping_scaler.pkl"),
                         ("finger_tapping_ordinal", "ordinal_finger_tapping_scaler.pkl")]:
            path = os.path.join(model_dir, fn)
            if os.path.exists(path):
                try:
                    self._scalers[name] = joblib.load(path)
                except:
                    pass

        gait_models = [("gait_rf", "rf_gait_scorer.pkl"), ("gait_xgb", "xgb_gait_scorer.pkl")]
        for key, fn in gait_models:
            path = os.path.join(model_dir, fn)
            if os.path.exists(path):
                try:
                    self._models[key] = joblib.load(path)
                except:
                    pass

        gait_scaler = os.path.join(model_dir, "gait_scaler.pkl")
        if os.path.exists(gait_scaler):
            try:
                self._scalers["gait"] = joblib.load(gait_scaler)
            except:
                pass

        for name, fn in [("finger_tapping", "finger_tapping_feature_cols.json"),
                         ("gait", "gait_feature_cols.json")]:
            path = os.path.join(model_dir, fn)
            if os.path.exists(path):
                with open(path) as f:
                    self._feature_cols[name] = json.load(f)

        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def has_model(self, task: str, model_type: str = "rf") -> bool:
        return f"{task}_{model_type}" in self._models

    def get_available_models(self) -> Dict[str, list]:
        result = {"finger_tapping": [], "gait": []}
        for key in self._models:
            if key.startswith("finger_tapping_"):
                result["finger_tapping"].append(key.replace("finger_tapping_", ""))
            elif key.startswith("gait_"):
                result["gait"].append(key.replace("gait_", ""))
        return result

    def _get_ft_feature_cols(self):
        return ["tapping_speed", "amplitude_mean", "amplitude_std", "amplitude_decrement",
            "first_half_amplitude", "second_half_amplitude", "opening_velocity_mean",
            "closing_velocity_mean", "peak_velocity_mean", "velocity_decrement",
            "rhythm_variability", "hesitation_count", "halt_count", "freeze_episodes",
            "fatigue_rate", "velocity_first_third", "velocity_mid_third", "velocity_last_third",
            "amplitude_first_third", "amplitude_mid_third", "amplitude_last_third",
            "velocity_slope", "amplitude_slope", "rhythm_slope",
            "variability_first_half", "variability_second_half", "variability_change"]

    def _get_gait_feature_cols(self):
        return ["arm_swing_amplitude_mean", "arm_swing_asymmetry", "walking_speed", "cadence",
            "step_height_mean", "step_count", "stride_length", "stride_variability",
            "swing_time_mean", "stance_time_mean", "swing_stance_ratio", "double_support_percent",
            "step_length_asymmetry", "swing_time_asymmetry", "trunk_flexion_mean", "trunk_flexion_rom",
            "hip_flexion_rom_mean", "knee_flexion_rom_mean", "ankle_dorsiflexion_rom_mean",
            "step_length_first_half", "step_length_second_half", "step_length_trend",
            "cadence_first_half", "cadence_second_half", "cadence_trend",
            "arm_swing_first_half", "arm_swing_second_half", "arm_swing_trend",
            "stride_variability_first_half", "stride_variability_second_half", "variability_trend",
            "step_height_first_half", "step_height_second_half", "step_height_trend"]

    def _engineer_ft_features(self, d: dict) -> dict:
        """Calculate engineered features for finger tapping (from train_finger_tapping_ml_v2.py)"""
        d = d.copy()
        total_taps = d.get('total_taps', 25)
        duration = d.get('duration', 10.0)
        amplitude_mean = d.get('amplitude_mean', 0.1)
        amplitude_std = d.get('amplitude_std', 0.01)
        hesitation_count = d.get('hesitation_count', 0)
        halt_count = d.get('halt_count', 0)
        amplitude_decrement = d.get('amplitude_decrement', 0)
        rhythm_variability = d.get('rhythm_variability', 0)
        fatigue_rate = d.get('fatigue_rate', 0)
        tapping_speed = d.get('tapping_speed', 2.0)

        # Engineered features
        d['tap_per_second'] = total_taps / (duration + 0.001)
        d['amplitude_cv'] = amplitude_std / (amplitude_mean + 0.001)
        d['event_ratio'] = (hesitation_count + halt_count) / (total_taps + 1)
        d['severity_index'] = (amplitude_decrement * 0.3 + rhythm_variability * 0.3 +
                               fatigue_rate * 0.2 + d['event_ratio'] * 0.2)
        d['speed_amplitude'] = tapping_speed * amplitude_mean
        d['amplitude_normalized'] = amplitude_mean / 0.3  # Normalized by typical max
        d['speed_normalized'] = tapping_speed / 4.0  # Normalized by typical max
        d['fatigue_severity'] = fatigue_rate * amplitude_decrement
        return d

    def _extract_features(self, metrics: Any, task: str) -> np.ndarray:
        feature_cols = self._feature_cols.get(task, [])
        if not feature_cols:
            feature_cols = self._get_ft_feature_cols() if task == "finger_tapping" else self._get_gait_feature_cols()

        if hasattr(metrics, "__dataclass_fields__"):
            metrics_dict = asdict(metrics)
        elif isinstance(metrics, dict):
            metrics_dict = metrics
        else:
            raise ValueError(f"Unknown metrics type: {type(metrics)}")

        # Apply feature engineering for finger tapping
        if task == "finger_tapping":
            metrics_dict = self._engineer_ft_features(metrics_dict)

        features = [float(metrics_dict.get(col, 0.0) or 0.0) for col in feature_cols]
        return np.array(features).reshape(1, -1)

    def predict_finger_tapping(self, metrics, model_type: str = "rf") -> Optional[MLPrediction]:
        if not self._loaded:
            self.load_models()
        model_key = f"finger_tapping_{model_type}"
        if model_key not in self._models:
            return None
        model = self._models[model_key]
        scaler_key = "finger_tapping_ordinal" if model_type == "ordinal" else "finger_tapping"
        scaler = self._scalers.get(scaler_key)
        if scaler is None:
            return None
        X = self._extract_features(metrics, "finger_tapping")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        raw_pred = model.predict(X_scaled)[0]
        if model_type == "ordinal" and hasattr(raw_pred, "__len__"):
            score = int(raw_pred)
            confidence = 0.8
        else:
            score = int(np.clip(np.round(raw_pred), 0, 4))
            confidence = 1.0 - abs(raw_pred - score)
        return MLPrediction(score=score, confidence=round(confidence, 3),
                           raw_prediction=round(float(raw_pred), 3), model_type=model_type)

    def predict_gait(self, metrics, model_type: str = "rf") -> Optional[MLPrediction]:
        if not self._loaded:
            self.load_models()
        model_key = f"gait_{model_type}"
        if model_key not in self._models:
            return None
        model = self._models[model_key]
        scaler = self._scalers.get("gait")
        if scaler is None:
            return None
        X = self._extract_features(metrics, "gait")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        raw_pred = model.predict(X_scaled)[0]
        score = int(np.clip(np.round(raw_pred), 0, 4))
        confidence = 1.0 - abs(raw_pred - score)
        return MLPrediction(score=score, confidence=round(confidence, 3),
                           raw_prediction=round(float(raw_pred), 3), model_type=model_type)

    def predict_lstm(self, landmarks: list, task_type: str) -> Optional[MLPrediction]:
        """
        Predict UPDRS score using LSTM model (time-series)

        Args:
            landmarks: List of landmark frames
            task_type: 'finger_tapping', 'hand_movement', 'gait', etc.

        Returns:
            MLPrediction or None if LSTM not available
        """
        try:
            from models.lstm_model import LSTMScorerFactory

            # Map task types
            if task_type in ['finger_tapping', 'hand_movement']:
                lstm_task = 'finger_tapping'
            else:
                lstm_task = 'gait'

            scorer = LSTMScorerFactory.get_scorer(lstm_task)
            raw_pred, confidence = scorer.predict(landmarks)

            score = int(np.clip(np.round(raw_pred), 0, 4))

            return MLPrediction(
                score=score,
                confidence=round(confidence, 3),
                raw_prediction=round(float(raw_pred), 3),
                model_type='lstm'
            )
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return None


def get_ml_scorer() -> MLScorer:
    return MLScorer()
