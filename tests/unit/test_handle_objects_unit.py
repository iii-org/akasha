import pytest

from akasha.helper import handle_objects

pytestmark = pytest.mark.unit


def test_handle_model_supports_ollama_default_base(monkeypatch):
    captured = {}

    class _FakeRemoteModel:
        def __init__(self, base_url, temperature, api_key, model_name, **kwargs):
            captured["base_url"] = base_url
            captured["temperature"] = temperature
            captured["api_key"] = api_key
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(handle_objects, "_get_env_var", lambda env_file="": {})
    monkeypatch.setattr(
        __import__("akasha.utils.models.remo", fromlist=["remote_model"]),
        "remote_model",
        _FakeRemoteModel,
    )

    handle_objects.handle_model(
        "ollama:llama3.1",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert captured["base_url"] == "http://localhost:11434"
    assert captured["api_key"] == "ollama"
    assert captured["model_name"] == "llama3.1"
    assert captured["kwargs"]["max_output_tokens"] == 256


def test_handle_model_supports_ollama_custom_base(monkeypatch):
    captured = {}

    class _FakeRemoteModel:
        def __init__(self, base_url, temperature, api_key, model_name, **kwargs):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["model_name"] = model_name

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {
            "OLLAMA_API_BASE": "http://env-host:11434",
            "OLLAMA_API_KEY": "secret",
        },
    )
    monkeypatch.setattr(
        __import__("akasha.utils.models.remo", fromlist=["remote_model"]),
        "remote_model",
        _FakeRemoteModel,
    )

    handle_objects.handle_model("ollama:http://custom-host:11434@qwen3:8b")

    assert captured["base_url"] == "http://custom-host:11434"
    assert captured["api_key"] == "secret"
    assert captured["model_name"] == "qwen3:8b"


def test_handle_model_requires_ollama_model_name():
    with pytest.raises(ValueError):
        handle_objects.handle_model("ollama:")
