"""
Tests for benchmarking a vision-language model on freezing-of-gait timing.
"""

import pytest

from services.fog_benchmark import (
    DataUseNotApproved,
    build_prediction_record,
    build_prompt,
    parse_freezing_response,
    require_data_use_approval,
)


def test_prompt_carries_the_timebase_the_model_cannot_see():
    prompt = build_prompt(duration_sec=20.0, n_frames=10)
    assert "20.0 seconds" in prompt
    assert "20.0 / 10" in prompt
    # the clinical distinction that makes the task non-trivial
    assert "do NOT count" in prompt


@pytest.mark.parametrize("duration, frames", [(0, 10), (20, 0), (-1, 5)])
def test_prompt_rejects_impossible_timebase(duration, frames):
    with pytest.raises(ValueError):
        build_prompt(duration, frames)


def test_parses_plain_json():
    text = '{"freezing_intervals": [{"start": 3.2, "end": 9.5, "note": "발이 붙음"}]}'
    assert parse_freezing_response(text) == [{"start": 3.2, "end": 9.5, "note": "발이 붙음"}]


def test_parses_fenced_json_and_surrounding_prose():
    text = "Here is what I observed:\n```json\n{\"freezing_intervals\": [{\"start\": 1, \"end\": 4}]}\n```\nHope that helps."
    assert parse_freezing_response(text) == [{"start": 1.0, "end": 4.0}]


def test_empty_list_means_no_freezing_not_a_failure():
    assert parse_freezing_response('{"freezing_intervals": []}') == []


def test_unparseable_replies_yield_nothing_rather_than_guesses():
    # a benchmark that invents intervals when the model rambles would flatter it
    for text in ["I could not tell.", "", None, "{broken json", '["not", "a", "dict"]']:
        assert parse_freezing_response(text) == []


def test_malformed_intervals_are_dropped():
    text = """{"freezing_intervals": [
        {"start": 5, "end": 3},
        {"start": -2, "end": 4},
        {"start": "abc", "end": 9},
        {"start": 2, "end": 6}
    ]}"""
    assert parse_freezing_response(text) == [{"start": 2.0, "end": 6.0}]


def test_intervals_are_clipped_to_clip_duration():
    text = '{"freezing_intervals": [{"start": 8, "end": 30}, {"start": 25, "end": 28}]}'
    # the second interval starts past the end of the clip and is dropped
    assert parse_freezing_response(text, duration_sec=20) == [{"start": 8.0, "end": 20.0}]


def test_timestamp_strings_are_accepted():
    text = '{"freezing_intervals": [{"start": "1:05", "end": "1:12"}]}'
    assert parse_freezing_response(text) == [{"start": 65.0, "end": 72.0}]


def test_intervals_come_back_in_time_order():
    text = '{"freezing_intervals": [{"start": 9, "end": 11}, {"start": 2, "end": 4}]}'
    assert [i["start"] for i in parse_freezing_response(text)] == [2.0, 9.0]


def test_prediction_record_matches_the_evaluator_shape():
    record = build_prediction_record(
        "PD4T_S12_gait_03",
        [{"start": 3.0, "end": 9.0}],
        model="gpt-4o",
        split="test",
    )
    assert record["gait_events"]["freezing_intervals"] == [{"start": 3.0, "end": 9.0}]
    assert record["primitives"]["freezing_of_gait"]["severity"] == 1
    assert record["primitives"]["freezing_of_gait"]["evidence"] == [{"start_sec": 3.0, "end_sec": 9.0}]
    assert record["condition"] == "commercial_zero_shot"
    assert record["split"] == "test"


def test_no_freezing_records_severity_zero_with_no_evidence():
    record = build_prediction_record("c1", [], model="gemini-2.0-flash")
    assert record["primitives"]["freezing_of_gait"]["severity"] == 0
    assert record["primitives"]["freezing_of_gait"]["evidence"] == []
    assert record["gait_events"]["freezing_intervals"] == []


def test_third_party_upload_is_blocked_without_explicit_approval():
    with pytest.raises(DataUseNotApproved) as exc:
        require_data_use_approval(False, "the OpenAI API")
    assert "third party" in str(exc.value)
    # approval is a caller decision, not a default
    require_data_use_approval(True, "the OpenAI API")
