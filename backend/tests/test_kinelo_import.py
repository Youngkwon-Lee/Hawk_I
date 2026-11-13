"""
Round-trip Kinelo's hawkeye-training/v1 golden export through the Hawk_I loader.

The fixture is synthetic (no patient data) and is copied verbatim from
kinelo-research-prototype server/fixtures/hawkeye_training_v1.golden.jsonl.
"""

import json
from pathlib import Path

import pytest

from services.kinelo_import import (
    TRAINABLE_PRIMITIVES,
    from_kinelo_record,
    load_kinelo_export,
    unassessable_clips,
)
from services.label_export import patient_overlap, summarise, to_training_record

FIXTURE = Path(__file__).parent / "fixtures" / "hawkeye_training_v1.golden.jsonl"


def _raw_lines():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_loads_every_golden_line():
    records = load_kinelo_export(FIXTURE)
    assert [r["clip_id"] for r in records] == [line["clip_id"] for line in _raw_lines()]


def test_records_match_to_training_record_shape():
    reference = to_training_record({"clip_id": "x"}, "train")
    for record in load_kinelo_export(FIXTURE):
        assert set(reference) <= set(record)
        assert set(record["primitives"]) == set(reference["primitives"])


def test_no_primitive_flips_between_null_and_value():
    for raw, record in zip(_raw_lines(), load_kinelo_export(FIXTURE)):
        for name in TRAINABLE_PRIMITIVES:
            source = raw["primitives"].get(name)
            loaded = record["primitives"][name]
            if source is None:
                # not observable / not rated must never become 0
                assert loaded is None, (raw["clip_id"], name)
            else:
                assert loaded == float(source) and isinstance(loaded, float), (raw["clip_id"], name)


def test_turning_impairment_is_dropped():
    for record in load_kinelo_export(FIXTURE):
        assert "turning_impairment" not in record["primitives"]


def test_score_subject_and_provenance():
    by_clip = {r["clip_id"]: r for r in load_kinelo_export(FIXTURE)}
    assert by_clip["synthetic-clip-1"]["updrs_3_10"] == 1
    assert by_clip["synthetic-clip-3"]["updrs_3_10"] is None
    assert by_clip["synthetic-clip-1"]["patient_id"] == "SYN01"
    assert by_clip["synthetic-clip-4"]["patient_id"] == ""

    adjudicated = by_clip["synthetic-clip-2"]
    assert adjudicated["adjudicated"] is True
    assert adjudicated["adjudicator_id"] == "adjudicator-1"
    assert adjudicated["annotators"] == ["rater-a", "rater-b"]
    assert adjudicated["schema_sha256"] == "4ea31895141d172a260bf12009348156c8fec94dfdeea3b1fb16f28ca10ebce0"
    assert adjudicated["media_sha256"] == "2" * 64
    assert adjudicated["split"] == "validation"


def test_null_intervals_become_empty_and_unassessable_is_reported():
    records = load_kinelo_export(FIXTURE)
    by_clip = {r["clip_id"]: r for r in records}
    assert by_clip["synthetic-clip-3"]["turning_intervals"] == []
    assert by_clip["synthetic-clip-3"]["freezing_intervals"] == []
    assert by_clip["synthetic-clip-4"]["freezing_intervals"] == []
    assert by_clip["synthetic-clip-1"]["turning_intervals"] == [
        {"start": 1.2, "end": 2.8},
        {"start": 5.0, "end": 6.4},
    ]
    assert unassessable_clips(records) == ["synthetic-clip-3"]


def test_summarise_and_patient_overlap():
    records = load_kinelo_export(FIXTURE)
    assert patient_overlap(records) == {}
    summary = summarise(records)
    assert {split: bucket["clips"] for split, bucket in summary.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }


def test_wrong_format_is_rejected(tmp_path):
    line = _raw_lines()[0]
    line["format"] = "hawkeye-training/v2"
    with pytest.raises(ValueError, match="unsupported export format"):
        from_kinelo_record(line)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(FIXTURE.read_text(encoding="utf-8") + json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:5: unsupported export format"):
        load_kinelo_export(bad)


def test_missing_format_is_rejected():
    line = _raw_lines()[0]
    del line["format"]
    with pytest.raises(ValueError, match="unsupported export format"):
        from_kinelo_record(line)
