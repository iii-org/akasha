"""Real provider contract smoke tests.

These tests intentionally cross the provider adapter boundary.  They are
opt-in because they consume provider quota or require a reachable Ollama
server.  They do not exercise RAG, embeddings, or MCP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values, load_dotenv
from tests.support.paths import REPO_ROOT, TEST_ENV_FILE

import akasha


ENV_FILE = TEST_ENV_FILE
MODEL_MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"
RUN_LIVE = os.getenv("RUN_PROVIDER_SMOKE", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

REQUIRED_KEYS = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}

_manifest = yaml.safe_load(MODEL_MANIFEST.read_text(encoding="utf-8"))
PROVIDER_MODELS = [
    (item["provider"], item["id"], REQUIRED_KEYS.get(item["provider"]))
    for item in _manifest["models"]
]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_PROVIDER_SMOKE=1 to enable real provider smoke tests",
    ),
]


def _load_test_env() -> str:
    if ENV_FILE.exists():
        # Permit the runner to select the endpoint without modifying the
        # local secrets file (for example public OpenAI vs Azure-compatible).
        load_dotenv(ENV_FILE, override=False)
    # Use process environment after loading the file so the runner can select
    # the public OpenAI endpoint without writing a second secrets file.
    return ""


def _skip_if_provider_unconfigured(required_key: str | None) -> None:
    if required_key is None:
        return
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if not os.getenv(required_key) and not values.get(required_key):
        pytest.skip(f"{required_key} is not configured")


def _assert_json_safe(value) -> None:
    json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_ask_non_stream_contract(provider, model, required_key):
    """A real provider must accept a short request and return visible text."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    qa = akasha.ask(
        model=model,
        stream=False,
        thinking=False,
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=128,
        env_file=env_file,
    )

    response = qa("Reply with exactly: PROVIDER_SMOKE_OK")

    assert isinstance(response, str), f"{provider} returned {type(response).__name__}"
    assert response.strip(), f"{provider} returned an empty response"
    assert qa.response == response
    assert qa.logs
    _assert_json_safe(qa.logs)


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_ask_stream_contract(provider, model, required_key):
    """A real provider stream must stay an iterable of text chunks."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    qa = akasha.ask(
        model=model,
        stream=True,
        thinking=False,
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=128,
        env_file=env_file,
    )

    chunks = list(qa("Reply with exactly: PROVIDER_STREAM_OK"))

    assert chunks, f"{provider} returned no stream chunks"
    assert all(isinstance(chunk, str) for chunk in chunks), (
        f"{provider} stream yielded non-text values: "
        f"{[type(chunk).__name__ for chunk in chunks]}"
    )
    assert "".join(chunks).strip()


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_agents_non_stream_contract(provider, model, required_key):
    """The LangChain agent adapter must also receive a final AI message."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    agent = akasha.agents(
        model=model,
        stream=False,
        thinking=False,
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=128,
        env_file=env_file,
        max_round=2,
    )

    response = agent("Reply with exactly: AGENT_PROVIDER_SMOKE_OK")

    assert isinstance(response, str), f"{provider} returned {type(response).__name__}"
    assert response.strip(), f"{provider} returned an empty agent response"
    _assert_json_safe(agent.logs)


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_agents_stream_contract(provider, model, required_key):
    """Agent streaming must normalize real provider chunks into events."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    agent = akasha.agents(
        model=model,
        stream=True,
        thinking=False,
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=128,
        env_file=env_file,
        max_round=2,
    )

    events = list(agent("Reply with exactly: AGENT_STREAM_PROVIDER_SMOKE_OK"))

    assert events, f"{provider} returned no agent events"
    assert all(
        isinstance(event, dict) and event.get("type") in {"answer", "tool", "warning"}
        for event in events
    )
    answer = "".join(event.get("data", "") for event in events if event["type"] == "answer")
    assert answer.strip(), f"{provider} returned no answer event"
    _assert_json_safe(events)
    _assert_json_safe(agent.logs)


def test_gemini_thinking_true_ask_contract():
    """Gemini's native thinking path must preserve answer/thinking separation."""
    provider, model, required_key = next(
        item for item in PROVIDER_MODELS if item[0] == "gemini"
    )
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    qa = akasha.ask(
        model=model,
        stream=True,
        thinking=True,
        thinking_budget="medium",
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=256,
        env_file=env_file,
    )

    events = list(qa("Think briefly, then answer exactly: THINKING_SMOKE_OK"))

    assert any(event["type"] == "answer" for event in events)
    assert "".join(
        event["data"] for event in events if event["type"] == "answer"
    ).strip()
    assert all(event["type"] in {"thinking", "answer"} for event in events)
    _assert_json_safe(events)


def test_gemini_thinking_true_agents_contract():
    """Gemini agents must expose native thinking events without corrupting answer."""
    provider, model, required_key = next(
        item for item in PROVIDER_MODELS if item[0] == "gemini"
    )
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    agent = akasha.agents(
        model=model,
        stream=True,
        thinking=True,
        thinking_budget="medium",
        keep_logs=True,
        temperature=0.0,
        max_output_tokens=256,
        env_file=env_file,
        max_round=2,
    )

    events = list(agent("Think briefly, then answer exactly: AGENT_THINKING_SMOKE_OK"))

    assert any(event["type"] == "answer" for event in events)
    assert "".join(
        event["data"] for event in events if event["type"] == "answer"
    ).strip()
    assert all(event["type"] in {"thinking", "answer", "tool"} for event in events)
    _assert_json_safe(events)
    _assert_json_safe(agent.logs)


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_thinking_true_ask_contract(provider, model, required_key):
    """Each configured provider must receive its native thinking configuration."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()

    qa = akasha.ask(
        model=model,
        thinking=True,
        thinking_budget="medium",
        max_output_tokens=256,
        env_file=env_file,
        keep_logs=True,
    )
    response = qa("Think briefly, then answer exactly: PROVIDER_THINKING_OK")

    assert isinstance(response, str) and response.strip()
    _assert_json_safe(qa.logs)


@pytest.mark.parametrize("provider,model,required_key", PROVIDER_MODELS)
def test_provider_thinking_true_stream_contract(provider, model, required_key):
    """Thinking-enabled streaming must still yield a normalized answer."""
    _skip_if_provider_unconfigured(required_key)
    env_file = _load_test_env()
    qa = akasha.ask(
        model=model,
        stream=True,
        thinking=True,
        thinking_budget="medium",
        max_output_tokens=256,
        env_file=env_file,
        keep_logs=True,
    )
    events = list(qa("Think briefly, then answer exactly: PROVIDER_THINKING_STREAM_OK"))

    assert any(event["type"] == "answer" for event in events)
    assert "".join(
        event["data"] for event in events if event["type"] == "answer"
    ).strip()
    assert all(event["type"] in {"thinking", "answer"} for event in events)
    _assert_json_safe(events)
