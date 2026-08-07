"""
Tests for turning workbench labels into a training-ready export.
"""

from services.label_export import (
    TRAINABLE_FIELDS,
    patient_overlap,
    summarise,
    to_training_record,
)


def _label(**overrides):
    base = {
        "clip_id": "14-006028",
        "patient_id": "42",
        "source_clip_path": "D:/clip/gait/042/14-006028_042_clip.mp4",
        "updrs_score": "2",
        "gait_speed": "경미↓",
        "stride_length": "정상",
        "left_right_asymmetry": "mild",
        "arm_swing_asymmetry": "none",
        "festination": "none",
        "freezing": "",
        "turning_quality": "",
        "stooped_posture": "mild",
        "postural_stability": "normal",
        "annotator": "dongsung",
        "confidence": "high",
        "gait_events": {
            "turning_intervals": [{"start": 8.7, "end": 9.4}],
            "freezing_intervals": [{"start": 14.9, "end": 19.0}],
        },
    }
    base.update(overrides)
    return base


def test_text_severities_become_numbers():
    record = to_training_record(_label(), "test")
    assert record["updrs_3_10"] == 2
    assert record["primitives"]["gait_speed_reduction"] == 1.0
    assert record["primitives"]["shortened_stride"] == 0.0
    assert record["primitives"]["step_length_asymmetry"] == 1.0
    assert record["primitives"]["trunk_flexion"] == 1.0
    assert record["primitives"]["postural_instability"] == 0.0


def test_blank_severity_stays_null_and_never_becomes_zero():
    record = to_training_record(_label(), "test")
    # an occluded finding must not train the model as if it were healthy
    assert record["primitives"]["freezing_of_gait"] is None


def test_turning_quality_is_excluded_from_trainable_fields():
    # blank in 99.6% of training clips, so it carries no signal
    assert "turning_quality" not in TRAINABLE_FIELDS
    record = to_training_record(_label(), "train")
    assert "turning_impairment" not in record["primitives"]


def test_turning_spans_are_kept_even_though_its_severity_is_dropped():
    record = to_training_record(_label(), "train")
    assert record["turning_intervals"] == [{"start": 8.7, "end": 9.4}]
    assert record["freezing_intervals"] == [{"start": 14.9, "end": 19.0}]


def test_malformed_spans_are_dropped_and_order_is_stable():
    label = _label(gait_events={"turning_intervals": [
        {"start": 9, "end": 5},
        {"start": "x", "end": 3},
        "junk",
        {"start": 4, "end": 6},
        {"start": 1, "end": 2},
    ]})
    assert to_training_record(label, "train")["turning_intervals"] == [
        {"start": 1.0, "end": 2.0}, {"start": 4.0, "end": 6.0}
    ]


def test_missing_score_stays_null():
    assert to_training_record(_label(updrs_score=""), "test")["updrs_3_10"] is None


def test_summary_reports_class_balance_and_coverage():
    records = [
        to_training_record(_label(updrs_score="0", patient_id="1"), "train"),
        to_training_record(_label(updrs_score="0", patient_id="1"), "train"),
        to_training_record(_label(updrs_score="2", patient_id="2"), "train"),
    ]
    summary = summarise(records)["train"]

    assert summary["clips"] == 3
    assert summary["patients"] == 2
    assert summary["score_counts"] == {"0": 2, "2": 1}
    assert summary["clips_with_freezing"] == 3
    # freezing severity is blank in the fixture, so coverage stays at zero
    assert summary["primitive_filled"]["freezing_of_gait"] == 0
    assert summary["primitive_filled"]["gait_speed_reduction"] == 3


def test_patient_overlap_between_splits_is_reported():
    records = [
        to_training_record(_label(patient_id="7"), "train"),
        to_training_record(_label(patient_id="7"), "test"),
        to_training_record(_label(patient_id="9"), "validation"),
    ]
    assert patient_overlap(records) == {"test|train": ["7"]}


def test_clean_split_reports_no_overlap():
    records = [
        to_training_record(_label(patient_id="1"), "train"),
        to_training_record(_label(patient_id="2"), "test"),
    ]
    assert patient_overlap(records) == {}
