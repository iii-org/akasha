import pytest

from akasha.helper import handle_objects

pytestmark = pytest.mark.unit


def test_handle_model_supports_ollama_default_base(monkeypatch):
    captured = {}

    def fake_build(
        provider, model_name, env, temperature, max_output_tokens,
        thinking=False, thinking_budget=None,
    ):
        captured.update(
            provider=provider,
            model_name=model_name,
            env=env,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return object()

    monkeypatch.setattr(handle_objects, "_get_env_var", lambda env_file="": {})
    monkeypatch.setattr(
        "akasha.utils.models.chat.build_chat_model", fake_build
    )

    handle_objects.handle_model(
        "ollama:llama3.1",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert captured["provider"] == "ollama"
    assert captured["env"]["OLLAMA_API_BASE"] == "http://localhost:11434"
    assert captured["model_name"] == "llama3.1"
    assert captured["max_output_tokens"] == 256


def test_handle_model_supports_ollama_custom_base(monkeypatch):
    captured = {}

    def fake_build(
        provider, model_name, env, temperature, max_output_tokens,
        thinking=False, thinking_budget=None,
    ):
        captured.update(provider=provider, model_name=model_name, env=env)
        return object()

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {"OLLAMA_API_BASE": "http://env-host:11434"},
    )
    monkeypatch.setattr(
        "akasha.utils.models.chat.build_chat_model", fake_build
    )

    handle_objects.handle_model("ollama:http://custom-host:11434@qwen3:8b")

    assert captured["provider"] == "ollama"
    assert captured["env"]["OLLAMA_API_BASE"] == "http://custom-host:11434"
    assert captured["model_name"] == "qwen3:8b"


def test_handle_model_requires_ollama_model_name():
    with pytest.raises(ValueError):
        handle_objects.handle_model("ollama:")
