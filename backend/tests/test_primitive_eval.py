"""
Tests for grading structured observations against human labels.
"""

from services import primitive_eval
from services.primitive_eval import (
    extract_label_primitives,
    extract_predicted_primitives,
    label_distribution,
    per_primitive_agreement,
    score_vs_observation,
    spearman,
    top_finding,
)


def test_label_fields_map_to_ontology_primitives():
    label = {
        "updrs_score": 2,
        "gait_speed": 2,
        "stride_length": 1,
        "freezing": 0,
        "turning_quality": "",
        "unrelated_field": 3,
    }
    assert extract_label_primitives(label) == {
        "gait_speed_reduction": 2.0,
        "shortened_stride": 1.0,
        "freezing_of_gait": 0.0,
    }


def test_unobserved_predictions_are_excluded_not_treated_as_zero():
    prediction = {
        "primitives": {
            "gait_speed_reduction": {"observability": "observed", "severity": 2},
            "freezing_of_gait": {"observability": "unobservable", "severity": None},
            "festination": {"observability": "uncertain", "severity": None},
        }
    }
    assert extract_predicted_primitives(prediction) == {"gait_speed_reduction": 2.0}


def test_per_primitive_agreement_reports_each_separately():
    pairs = [
        ({"gait_speed_reduction": 2, "freezing_of_gait": 0}, {"gait_speed_reduction": 2, "freezing_of_gait": 0}),
        ({"gait_speed_reduction": 1, "freezing_of_gait": 0}, {"gait_speed_reduction": 2, "freezing_of_gait": 0}),
        ({"gait_speed_reduction": 3, "freezing_of_gait": 2}, {"gait_speed_reduction": 3, "freezing_of_gait": 0}),
    ]
    report = per_primitive_agreement(pairs)

    assert report["gait_speed_reduction"]["n"] == 3
    assert report["gait_speed_reduction"]["exact_match"] == round(2 / 3, 4)
    assert report["gait_speed_reduction"]["within_1"] == 1.0
    # freezing is graded on its own, so a strong speed result cannot mask it
    assert report["freezing_of_gait"]["exact_match"] == round(2 / 3, 4)


def test_agreement_skips_primitives_the_model_did_not_observe():
    pairs = [({"gait_speed_reduction": 2, "festination": 1}, {"gait_speed_reduction": 2})]
    report = per_primitive_agreement(pairs)
    assert "festination" not in report
    assert report["gait_speed_reduction"]["n"] == 1


def test_distribution_exposes_the_always_zero_trap():
    pairs = [({"freezing_of_gait": 0}, {}) for _ in range(19)]
    pairs.append(({"freezing_of_gait": 2}, {}))

    report = label_distribution(pairs)["freezing_of_gait"]

    assert report["n"] == 20
    assert report["zero_share"] == 0.95
    # a model doing nothing at all would score 95% here
    assert report["always_zero_accuracy"] == 0.95
    assert report["majority_accuracy"] == 0.95


def test_top_finding_picks_most_severe_and_is_none_when_all_normal():
    assert top_finding({"gait_speed_reduction": 1, "festination": 3}) == "festination"
    assert top_finding({"gait_speed_reduction": 0, "festination": 0}) is None
    # deterministic tie-break by name
    assert top_finding({"b_primitive": 2, "a_primitive": 2}) == "a_primitive"


def test_cross_table_separates_lucky_hits_from_real_ones():
    records = [
        {   # right score, right reason
            "true_score": 2, "predicted_score": 2,
            "label_primitives": {"gait_speed_reduction": 2, "festination": 0},
            "predicted_primitives": {"gait_speed_reduction": 2, "festination": 0},
        },
        {   # right score, wrong reason - the cell score-only evaluation cannot see
            "true_score": 2, "predicted_score": 2,
            "label_primitives": {"gait_speed_reduction": 2, "festination": 0},
            "predicted_primitives": {"gait_speed_reduction": 0, "festination": 2},
        },
        {   # wrong score, right reason
            "true_score": 3, "predicted_score": 1,
            "label_primitives": {"festination": 3},
            "predicted_primitives": {"festination": 1},
        },
        {   # wrong score, wrong reason
            "true_score": 3, "predicted_score": 0,
            "label_primitives": {"festination": 3},
            "predicted_primitives": {"trunk_flexion": 1},
        },
    ]

    table = score_vs_observation(records)

    assert table["score_right_observation_right"] == 1
    assert table["score_right_observation_wrong"] == 1
    assert table["score_wrong_observation_right"] == 1
    assert table["score_wrong_observation_wrong"] == 1
    assert table["evaluated"] == 4
    assert table["lucky_hit_share"] == 0.25


def test_cross_table_skips_records_missing_a_side():
    records = [
        {"true_score": 2, "predicted_score": None, "label_primitives": {"a": 1}, "predicted_primitives": {"a": 1}},
        {"true_score": 2, "predicted_score": 2, "label_primitives": {}, "predicted_primitives": {"a": 1}},
    ]
    table = score_vs_observation(records)
    assert table["skipped_missing_data"] == 2
    assert table["evaluated"] == 0
    assert table["lucky_hit_share"] is None


def test_score_tolerance_allows_within_one_grading():
    records = [{
        "true_score": 2, "predicted_score": 3,
        "label_primitives": {"festination": 2},
        "predicted_primitives": {"festination": 2},
    }]
    assert score_vs_observation(records)["score_wrong_observation_right"] == 1
    assert score_vs_observation(records, score_tolerance=1)["score_right_observation_right"] == 1


def test_spearman_handles_ties_and_undefined_cases():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # no variance on one side
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    # too few points
    assert spearman([1, 2], [2, 1]) is None


def test_gait_mapping_covers_all_nine_primitives():
    assert len(primitive_eval.GAIT_LABEL_TO_PRIMITIVE) == 9
