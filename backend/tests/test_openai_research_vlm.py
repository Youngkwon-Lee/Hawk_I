from types import SimpleNamespace

from services.openai_research_vlm import (
    GPT56TerraConfig,
    GPT56TerraResearchVLM,
    get_config,
)


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=(
                '{"task_confirmed":true,"speed":"slow","amplitude":"small",'
                '"rhythm":"regular","pauses_or_hesitations":"none observed",'
                '"laterality_or_asymmetry":"not assessable",'
                '"overall_observation":"sample observation",'
                '"limitations":"sampled frames only"}'
            )
        )


def test_config_is_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_config() is None


def test_config_uses_bounded_safe_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GPT56_TERRA_MAX_FRAMES", "99")
    monkeypatch.delenv("GPT56_TERRA_MODEL", raising=False)
    config = get_config()
    assert config is not None
    assert config.model == "gpt-5.6-terra"
    assert config.sample_fps == 5.0
    assert config.max_frames == 99
    assert config.reasoning_effort == "none"


def test_status_never_includes_api_key():
    service = GPT56TerraResearchVLM(
        config=GPT56TerraConfig("test-key", "gpt-5.6-terra", 5.0, 100, "none"),
        client=object(),
    )
    status = service.status()
    assert status["available"] is True
    assert status["input_mode"] == "timestamped_sampled_frames"
    assert status["sample_fps"] == 5.0
    assert "api_key" not in status


def test_analyze_uses_structured_frames_only(tmp_path, monkeypatch):
    video_path = tmp_path / "example.mp4"
    video_path.write_bytes(b"not-used-by-mock")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    service = GPT56TerraResearchVLM(
        config=GPT56TerraConfig("test-key", "gpt-5.6-terra", 5.0, 100, "none"),
        client=client,
    )
    monkeypatch.setattr(
        service,
        "_extract_frames",
        lambda *_: ([("frame-a", 0.0), ("frame-b", 0.2)], 0.4),
    )
    monkeypatch.setattr(service, "_encode_frame", lambda frame: frame)

    result = service.analyze_video(str(video_path), "finger_tapping")

    assert result["success"] is True
    assert result["frames_analyzed"] == 2
    assert result["sample_fps"] == 5.0
    assert result["observation"]["speed"] == "slow"
    assert responses.kwargs["model"] == "gpt-5.6-terra"
    assert responses.kwargs["reasoning"] == {"effort": "none"}
    assert responses.kwargs["text"]["format"]["type"] == "json_schema"
    content = responses.kwargs["input"][0]["content"]
    assert [item["type"] for item in content] == [
        "input_text", "input_text", "input_image", "input_text", "input_image"
    ]


def test_analyze_rejects_unstructured_output(tmp_path, monkeypatch):
    video_path = tmp_path / "example.mp4"
    video_path.write_bytes(b"not-used-by-mock")
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: SimpleNamespace(output_text="not-json"))
    )
    service = GPT56TerraResearchVLM(
        config=GPT56TerraConfig("test-key", "gpt-5.6-terra", 5.0, 100, "none"),
        client=client,
    )
    monkeypatch.setattr(service, "_extract_frames", lambda *_: ([("frame-a", 0.0)], 0.2))
    monkeypatch.setattr(service, "_encode_frame", lambda frame: frame)

    result = service.analyze_video(str(video_path), "finger_tapping")

    assert result == {
        "success": False,
        "error": "GPT-5.6 Terra returned an unstructured observation.",
    }
