"""
Unit Tests for Hawkeye Backend Services
"""
import pytest
import numpy as np


class TestMetricsCalculator:
    """Tests for MetricsCalculator service"""

    def test_finger_tapping_metrics_calculation(self, mock_finger_tapping_landmarks):
        """Test finger tapping metrics are calculated correctly"""
        from services.metrics_calculator import MetricsCalculator

        calc = MetricsCalculator(fps=30.0)
        metrics = calc.calculate_finger_tapping_metrics(mock_finger_tapping_landmarks)

        # Basic sanity checks
        assert metrics.tapping_speed > 0, "Tapping speed should be positive"
        assert metrics.total_taps >= 0, "Total taps should be non-negative"
        assert metrics.duration > 0, "Duration should be positive"
        assert 0 <= metrics.amplitude_mean <= 2, "Amplitude should be reasonable"

    def test_finger_tapping_insufficient_data(self):
        """Test that insufficient data raises ValueError"""
        from services.metrics_calculator import MetricsCalculator

        calc = MetricsCalculator(fps=30.0)

        # Only 5 frames - should fail
        short_frames = [
            {"frame_number": i, "timestamp": i / 30.0, "keypoints": [
                {"id": 0, "x": 0, "y": 0, "z": 0},
                {"id": 4, "x": 0, "y": 0, "z": 0},
                {"id": 5, "x": 0, "y": 0, "z": 0},
                {"id": 6, "x": 0, "y": 0, "z": 0},
                {"id": 7, "x": 0, "y": 0, "z": 0},
                {"id": 8, "x": 0.1, "y": 0.1, "z": 0.1},
            ]}
            for i in range(5)
        ]

        with pytest.raises(ValueError, match="Insufficient"):
            calc.calculate_finger_tapping_metrics(short_frames)

    def test_gait_metrics_calculation(self, mock_gait_landmarks):
        """Test gait metrics are calculated correctly"""
        from services.metrics_calculator import MetricsCalculator

        calc = MetricsCalculator(fps=30.0)
        metrics = calc.calculate_gait_metrics(mock_gait_landmarks)

        # Basic sanity checks
        assert metrics.walking_speed >= 0, "Walking speed should be non-negative"
        assert metrics.cadence >= 0, "Cadence should be non-negative"
        assert metrics.step_count >= 0, "Step count should be non-negative"
        assert metrics.duration > 0, "Duration should be positive"


class TestMLScorer:
    """Tests for ML Scorer service"""

    def test_ml_scorer_singleton(self):
        """Test MLScorer follows singleton pattern"""
        from services.ml_scorer import MLScorer

        scorer1 = MLScorer()
        scorer2 = MLScorer()
        assert scorer1 is scorer2, "MLScorer should be singleton"

    def test_load_models(self):
        """Test ML models can be loaded"""
        from services.ml_scorer import get_ml_scorer

        scorer = get_ml_scorer()
        result = scorer.load_models()

        assert result is True, "Models should load successfully"
        assert scorer.is_loaded(), "Scorer should be marked as loaded"

    def test_available_models(self):
        """Test available models are reported correctly"""
        from services.ml_scorer import get_ml_scorer

        scorer = get_ml_scorer()
        scorer.load_models()

        available = scorer.get_available_models()
        assert "finger_tapping" in available
        assert "gait" in available
        assert len(available["finger_tapping"]) > 0

    def test_predict_finger_tapping_rf(self, mock_finger_tapping_metrics):
        """Test RF prediction for finger tapping"""
        from services.ml_scorer import get_ml_scorer

        scorer = get_ml_scorer()
        scorer.load_models()

        result = scorer.predict_finger_tapping(mock_finger_tapping_metrics, model_type="rf")

        assert result is not None, "Prediction should not be None"
        assert 0 <= result.score <= 4, "Score should be 0-4"
        assert 0 <= result.confidence <= 1, "Confidence should be 0-1"
        assert result.model_type == "rf"

    def test_predict_finger_tapping_xgb(self, mock_finger_tapping_metrics):
        """Test XGB prediction for finger tapping"""
        from services.ml_scorer import get_ml_scorer

        scorer = get_ml_scorer()
        scorer.load_models()

        result = scorer.predict_finger_tapping(mock_finger_tapping_metrics, model_type="xgb")

        assert result is not None, "Prediction should not be None"
        assert 0 <= result.score <= 4, "Score should be 0-4"
        assert result.model_type == "xgb"


class TestUPDRSScorer:
    """Tests for UPDRS Scorer service"""

    def test_rule_based_finger_tapping_scoring(self, mock_finger_tapping_landmarks):
        """Test rule-based scoring for finger tapping"""
        from services.metrics_calculator import MetricsCalculator
        from services.updrs_scorer import UPDRSScorer

        # Calculate metrics first
        calc = MetricsCalculator(fps=30.0)
        metrics = calc.calculate_finger_tapping_metrics(mock_finger_tapping_landmarks)

        # Score with rule-based method (method passed to constructor)
        scorer = UPDRSScorer(method="rule")
        result = scorer.score_finger_tapping(metrics)

        # UPDRSScore is a dataclass, access attributes directly
        assert hasattr(result, "total_score")
        assert 0 <= result.total_score <= 4
        assert hasattr(result, "severity")
        assert result.method == "rule"

    def test_ensemble_scoring(self, mock_finger_tapping_landmarks):
        """Test ensemble scoring combines rule and ML"""
        from services.metrics_calculator import MetricsCalculator
        from services.updrs_scorer import UPDRSScorer

        calc = MetricsCalculator(fps=30.0)
        metrics = calc.calculate_finger_tapping_metrics(mock_finger_tapping_landmarks)

        # Ensemble method passed to constructor
        scorer = UPDRSScorer(method="ensemble")
        result = scorer.score_finger_tapping(metrics)

        # UPDRSScore is a dataclass
        assert hasattr(result, "total_score")
        assert result.method == "ensemble"
        assert hasattr(result, "details")
        # Ensemble should have both rule and ml scores in details
        if "rule" in result.details and "ml" in result.details:
            assert result.details["rule"] is not None or result.details["ml"] is not None


class TestVisualizationService:
    """Tests for Visualization Service"""

    def test_visualization_service_initialization(self):
        """Test VisualizationService initializes correctly"""
        from services.visualization import VisualizationService

        viz = VisualizationService()
        assert viz is not None

    @pytest.mark.skip(reason="Requires actual video file for heatmap generation")
    def test_generate_motion_heatmap(self, mock_finger_tapping_landmarks, tmp_path):
        """Test motion heatmap generation - requires video file"""
        from services.visualization import VisualizationService

        viz = VisualizationService()
        output_path = str(tmp_path / "test_heatmap.jpg")

        result = viz.generate_motion_heatmap(
            mock_finger_tapping_landmarks,
            output_path=output_path,
            video_path=None
        )
        assert result is None or isinstance(result, str)
