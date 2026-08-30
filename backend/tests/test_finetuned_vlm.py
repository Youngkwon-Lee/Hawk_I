"""
Tests for calling a self-hosted fine-tuned VLM and reading back primitives.
"""

import pytest

from services import finetuned_vlm
from services.finetuned_vlm import (
    FinetunedVLMConfig,
    FinetunedVLMUnavailable,
    apply_to_analysis_response,
    build_prompt,
    get_config,
    parse_response,
    to_analysis_fields,
)
from services.vlm_training_contract import build_c0b_messages


def _config():
    return FinetunedVLMConfig(base_url="https://gpu.example/v1", model="qwen3-vl-4b-c3")


def _reply(**overrides):
    payload = {
        "primitives": {
            "gait_speed_reduction": {"observability": "observed", "severity": 2, "confidence": "medium"},
            "freezing_of_gait": {"observability": "unobservable", "severity": None, "confidence": "high"},
        },
        "turning_intervals": [{"start": 8.7, "end": 9.4}],
        "freezing_intervals": [],
        "updrs_3_10": 2,
        "summary": "보폭 감소와 팔 흔들림 저하가 관찰됨",
    }
    payload.update(overrides)
    import json
    return json.dumps(payload, ensure_ascii=False)


def test_config_is_none_until_both_url_and_model_are_set(monkeypatch):
    monkeypatch.delenv("HAWKEYE_VLM_BASE_URL", raising=False)
    monkeypatch.delenv("HAWKEYE_VLM_MODEL", raising=False)
    assert get_config() is None

    monkeypatch.setenv("HAWKEYE_VLM_BASE_URL", "https://gpu.example/v1")
    assert get_config() is None, "a URL without a model name is not usable"

    monkeypatch.setenv("HAWKEYE_VLM_MODEL", "qwen3-vl-4b")
    config = get_config()
    assert config.base_url == "https://gpu.example/v1"
    assert config.model == "qwen3-vl-4b"
    assert config.condition == "C0B"


def test_trailing_slash_is_stripped_and_local_endpoint_detected(monkeypatch):
    monkeypatch.setenv("HAWKEYE_VLM_BASE_URL", "http://127.0.0.1:8000/v1/")
    monkeypatch.setenv("HAWKEYE_VLM_MODEL", "local-model")
    config = get_config()
    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.is_local is True


def test_prompt_carries_timebase_and_clinical_exclusions():
    prompt = build_prompt(duration_sec=20.0, n_frames=16)
    assert "20.0 seconds" in prompt
    assert "20.0 / 16" in prompt
    # without the exclusion the task collapses into "detect slow walking"
    assert "NOT freezing" in prompt
    # the rule that keeps an unseen finding from reading as normal
    assert "do NOT report it as 0" in prompt


@pytest.mark.parametrize("duration, frames", [(0, 16), (20, 0)])
def test_prompt_rejects_impossible_timebase(duration, frames):
    with pytest.raises(ValueError):
        build_prompt(duration, frames)


def test_parses_primitives_intervals_and_score():
    parsed = parse_response(_reply(), duration_sec=20)
    assert parsed["primitives"]["gait_speed_reduction"]["severity"] == 2
    assert parsed["turning_intervals"] == [{"start": 8.7, "end": 9.4}]
    assert parsed["updrs_3_10"] == 2
    assert parsed["summary"].startswith("보폭")


def test_unobservable_finding_never_carries_a_severity():
    reply = _reply(primitives={
        "freezing_of_gait": {"observability": "unobservable", "severity": 0, "confidence": "high"},
    })
    # a model claiming 0 for something it could not see must not be believed
    assert parse_response(reply)["primitives"]["freezing_of_gait"]["severity"] is None


def test_unknown_primitive_names_and_bad_severities_are_dropped():
    reply = _reply(primitives={
        "not_a_primitive": {"observability": "observed", "severity": 1},
        "festination": {"observability": "observed", "severity": 9},
        "trunk_flexion": {"observability": "sideways", "severity": 1},
        "shortened_stride": {"observability": "observed", "severity": 1},
    })
    assert list(parse_response(reply)["primitives"]) == ["shortened_stride"]


def test_intervals_are_clipped_and_malformed_ones_dropped():
    reply = _reply(turning_intervals=[
        {"start": 5, "end": 3},
        {"start": 18, "end": 40},
        {"start": 25, "end": 28},
        {"start": 2, "end": 4},
    ])
    assert parse_response(reply, duration_sec=20)["turning_intervals"] == [
        {"start": 2.0, "end": 4.0}, {"start": 18.0, "end": 20.0}
    ]


def test_fenced_json_and_surrounding_prose_are_tolerated():
    text = "Sure:\n```json\n" + _reply() + "\n```\nHope that helps."
    assert parse_response(text)["updrs_3_10"] == 2


def test_training_prompt_answer_line_is_accepted():
    assert parse_response("reasoning omitted\nanswer: 2")["updrs_3_10"] == 2


def test_non_json_reply_raises_rather_than_guessing():
    for text in ["I cannot tell.", "", None, "{broken"]:
        with pytest.raises(FinetunedVLMUnavailable):
            parse_response(text)


def test_out_of_range_score_becomes_null():
    assert parse_response(_reply(updrs_3_10=9))["updrs_3_10"] is None


def test_analysis_fields_keep_score_separate_from_primitives():
    fields = to_analysis_fields(parse_response(_reply(), duration_sec=20), _config())

    assert fields["scoring_method"] == "finetuned_vlm"
    assert fields["ml_model_type"] == "qwen3-vl-4b-c3"
    assert fields["gait_events"]["turning_intervals"] == [{"start": 8.7, "end": 9.4}]
    assert fields["ai_interpretation"]["summary"].startswith("보폭")
    # the anchor is predicted, never summed from severities
    assert fields["updrs_score"]["total_score"] == 2.0
    assert fields["updrs_score"]["severity"] == "Mild"
    assert fields["updrs_score"]["details"]["source"] == "model_predicted"


def test_zero_finetuned_score_is_labeled_normal():
    fields = to_analysis_fields(parse_response(_reply(updrs_3_10=0)), _config())

    assert fields["updrs_score"]["total_score"] == 0.0
    assert fields["updrs_score"]["severity"] == "Normal"


def test_missing_score_leaves_updrs_out_entirely():
    fields = to_analysis_fields(parse_response(_reply(updrs_3_10=None)), _config())
    assert "updrs_score" not in fields


def test_finetuned_score_becomes_primary_and_rule_score_is_retained():
    response = {
        "scoring_method": "rule",
        "ml_model_type": "rule_based",
        "updrs_score": {"total_score": 1.5, "method": "rule"},
        "ai_interpretation": {
            "summary": "규칙 기반 분석은 중간 정도의 보행 변화를 추정했습니다.",
            "explanation": "규칙 기반 설명",
        },
    }
    finetuned = {
        "scoring_method": "finetuned_vlm",
        "ml_model_type": "hawkeye-c0b-seed42",
        "updrs_score": {
            "score": 0.0,
            "total_score": 0.0,
            "method": "finetuned_vlm",
            "details": {"source": "model_predicted"},
        },
    }

    apply_to_analysis_response(response, finetuned)

    assert response["scoring_method"] == "finetuned_vlm"
    assert response["ml_model_type"] == "hawkeye-c0b-seed42"
    assert response["updrs_score"]["total_score"] == 0.0
    assert response["updrs_score"]["method"] == "finetuned_vlm"
    assert response["updrs_score"]["details"]["pipeline_reference"] == {
        "score": 1.5,
        "method": "rule",
        "summary": "규칙 기반 분석은 중간 정도의 보행 변화를 추정했습니다.",
    }
    assert response["ai_interpretation"]["summary"] == (
        "미세조정 모델은 이번 보행의 UPDRS 3.10 점수를 0점(Normal)으로 추정했습니다."
    )
    assert "hawkeye-c0b-seed42" in response["ai_interpretation"]["explanation"]
    assert "중간 정도" not in response["ai_interpretation"]["summary"]


def test_finetuned_narrative_is_retained_when_model_supplies_one():
    response = {
        "updrs_score": {"total_score": 2.7, "method": "rule"},
        "ai_interpretation": {"summary": "규칙 기반 설명"},
    }
    finetuned = {
        "scoring_method": "finetuned_vlm",
        "ml_model_type": "hawkeye-c0b-seed42",
        "updrs_score": {
            "score": 1.0,
            "total_score": 1.0,
            "severity": "Slight",
            "method": "finetuned_vlm",
        },
        "ai_interpretation": {
            "summary": "모델이 직접 생성한 설명",
            "explanation": "모델 설명",
            "recommendations": [],
        },
    }

    apply_to_analysis_response(response, finetuned)

    assert response["ai_interpretation"]["summary"] == "모델이 직접 생성한 설명"


def test_missing_finetuned_score_leaves_pipeline_score_in_place():
    response = {"updrs_score": {"total_score": 2.0, "method": "rule"}}

    apply_to_analysis_response(response, {"scoring_method": "finetuned_vlm"})

    assert response["updrs_score"] == {"total_score": 2.0, "method": "rule"}


def test_analyze_returns_none_when_not_configured(monkeypatch):
    monkeypatch.setattr(finetuned_vlm, "get_config", lambda: None)
    assert finetuned_vlm.analyze(["frame"], 20.0) is None


def test_analyze_returns_none_when_endpoint_is_down(monkeypatch):
    monkeypatch.setattr(finetuned_vlm, "get_config", _config)

    def boom(*args, **kwargs):
        raise FinetunedVLMUnavailable("connection refused")

    monkeypatch.setattr(finetuned_vlm, "call_endpoint", boom)
    # a missing endpoint must not fail an otherwise good analysis
    assert finetuned_vlm.analyze(["frame"], 20.0) is None


def test_analyze_returns_fields_on_success(monkeypatch):
    monkeypatch.setattr(finetuned_vlm, "get_config", _config)
    monkeypatch.setattr(finetuned_vlm, "call_endpoint", lambda *a, **k: _reply())

    fields = finetuned_vlm.analyze(["frame"], 20.0)
    assert fields["primitives"]["gait_speed_reduction"]["severity"] == 2
    assert fields["scoring_method"] == "finetuned_vlm"


def test_api_key_is_optional_for_local_endpoints(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": _reply()}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(finetuned_vlm.requests, "post", fake_post)
    finetuned_vlm.call_endpoint(["frame"], 20.0, FinetunedVLMConfig(
        base_url="http://127.0.0.1:8000/v1", model="local"
    ))

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    assert captured["json"]["messages"][0] == build_c0b_messages()[0]
    user_content = captured["json"]["messages"][1]["content"]
    assert [block["type"] for block in user_content] == ["text", "text", "text", "image_url"]
    assert user_content[0]["text"].startswith("Scoring Item: MDS-UPDRS")
    assert user_content[2]["text"].startswith("Score the gait task")
    assert captured["json"]["temperature"] == 0
    assert captured["json"]["max_tokens"] == 16


def test_non_autonomous_condition_is_not_silently_used(monkeypatch):
    with pytest.raises(FinetunedVLMUnavailable, match="not wired for autonomous inference"):
        finetuned_vlm.call_endpoint(
            ["frame"],
            20.0,
            FinetunedVLMConfig(
                base_url="http://127.0.0.1:8000/v1",
                model="hawkeye-c3be",
                condition="C3BE",
            ),
        )


def test_frames_to_base64_skips_decoding_when_not_configured(monkeypatch):
    monkeypatch.setattr(finetuned_vlm, "get_config", lambda: None)
    # no endpoint means no reason to pay video decoding cost
    assert finetuned_vlm.frames_to_base64("/nonexistent.mp4") == ([], 0.0)


def test_frames_to_base64_returns_empty_for_unreadable_video(monkeypatch):
    monkeypatch.setattr(finetuned_vlm, "get_config", _config)
    frames, duration = finetuned_vlm.frames_to_base64("/definitely/not/a/video.mp4")
    assert frames == [] and duration == 0.0
