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


def test_text_severity_words_map_to_ordinals():
    from services.primitive_eval import parse_severity
    # 두 어휘 체계가 함께 쓰입니다
    assert parse_severity("정상") == 0.0
    assert parse_severity("경미↓") == 1.0
    assert parse_severity("중등↓") == 2.0
    assert parse_severity("심함↓") == 3.0
    assert parse_severity("none") == 0.0
    assert parse_severity("mild") == 1.0
    assert parse_severity("moderate") == 2.0
    assert parse_severity("severe") == 3.0
    assert parse_severity("normal") == 0.0
    assert parse_severity("observed") == 1.0
    assert parse_severity(2) == 2.0
    assert parse_severity("2") == 2.0


def test_blank_severity_is_none_not_zero():
    from services.primitive_eval import parse_severity
    # 온톨로지 규칙: 빈 값은 "관찰 안 됨"이지 정상이 아닙니다
    for blank in (None, "", "   ", "알수없음"):
        assert parse_severity(blank) is None


def test_real_label_shape_maps_to_primitives():
    label = {
        "gait_speed": "경미↓",
        "stride_length": "정상",
        "arm_swing_asymmetry": "moderate",
        "postural_stability": "normal",
        "turning_quality": "",
        "freezing": "",
    }
    assert extract_label_primitives(label) == {
        "gait_speed_reduction": 1.0,
        "shortened_stride": 0.0,
        "arm_swing_asymmetry": 2.0,
        "postural_instability": 0.0,
    }


def test_intervals_read_from_gait_events_and_bad_spans_dropped():
    from services.primitive_eval import extract_intervals
    record = {"gait_events": {
        "turning_intervals": [
            {"start": 8.7, "end": 16.9},
            {"start": 2.0, "end": 2.7},
            {"start": 5.0, "end": 4.0},
            {"start": None, "end": 3.0},
            "not-a-span",
        ],
        "freezing_intervals": [{"start": 11.3, "end": 19.8}],
    }}
    assert extract_intervals(record) == [(2.0, 2.7), (8.7, 16.9)]
    assert extract_intervals(record, "freezing_intervals") == [(11.3, 19.8)]
    assert extract_intervals({"label": {"gait_events": {"turning_intervals": [{"start": 1, "end": 2}]}}}) == [(1.0, 2.0)]
    assert extract_intervals({}) == []


def test_temporal_agreement_matches_by_midpoint_within_tolerance():
    from services.primitive_eval import temporal_event_agreement
    pairs = [
        ([(2.0, 2.7)], [(2.2, 2.9)]),   # 0.2s 차이 - 매칭
        ([(10.0, 10.7)], [(14.0, 14.7)]),  # 4s 차이 - 미매칭
        ([(5.0, 5.7)], []),             # 놓침
    ]
    report = temporal_event_agreement(pairs, tolerance_sec=1.0)

    assert report["labeled_events"] == 3
    assert report["predicted_events"] == 2
    assert report["matched"] == 1
    assert report["recall"] == round(1 / 3, 4)
    assert report["precision"] == 0.5
    assert report["mean_iou_of_matched"] is not None


def test_temporal_agreement_does_not_double_match_one_prediction():
    from services.primitive_eval import temporal_event_agreement
    report = temporal_event_agreement([([(2.0, 2.5), (2.1, 2.6)], [(2.05, 2.55)])], tolerance_sec=1.0)
    assert report["matched"] == 1


def test_temporal_agreement_handles_empty_input():
    from services.primitive_eval import temporal_event_agreement
    report = temporal_event_agreement([([], [])])
    assert report["matched"] == 0
    assert report["precision"] is None and report["recall"] is None and report["f1"] is None
