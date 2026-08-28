import pytest

from services.c0b_training_eval import parse_c0b_answer, score_predictions


def test_parse_uses_last_valid_answer():
    assert parse_c0b_answer("answer: 1\nreconsidered\nanswer: 3") == 3
    assert parse_c0b_answer("no grade") is None
    assert parse_c0b_answer("answer: 7") is None


def test_metrics_include_ordinal_and_imbalance_checks():
    metrics = score_predictions([0, 0, 1, 2, 4], [0, None, 1, 1, 4])
    assert metrics["parse_failures"] == 1
    assert metrics["exact"] == pytest.approx(0.8)
    assert metrics["within_1"] == 1.0
    assert metrics["mae"] == pytest.approx(0.2)
    assert metrics["majority_class"] == 0
    assert metrics["majority_exact_baseline"] == pytest.approx(0.4)
    assert metrics["gt2_sensitivity"] == pytest.approx(0.5)
    assert len(metrics["confusion_matrix"]) == 5
