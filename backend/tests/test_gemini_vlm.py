from services.gemini_vlm import GeminiResearchVLM, get_config, parse_observation


def test_config_is_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_config() is None


def test_config_uses_safe_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_VLM_MODEL", raising=False)
    config = get_config()
    assert config is not None
    assert config.model == "gemini-2.5-flash"
    assert config.upload_timeout_seconds == 120


def test_parse_observation_accepts_json_fence():
    parsed = parse_observation('```json\n{"tap_speed":"slow","rhythm":"regular"}\n```')
    assert parsed == {"tap_speed": "slow", "rhythm": "regular"}


def test_parse_observation_rejects_truncated_json():
    assert parse_observation('{"tap_speed":"slow"') is None


def test_status_never_includes_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    status = GeminiResearchVLM().status()
    assert "api_key" not in status
    assert status["model"] == "gemini-2.5-flash"
