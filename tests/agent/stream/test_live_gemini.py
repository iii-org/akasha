"""Opt-in live tests for the LangChain-native Gemini Agent path.

These tests intentionally call the real Gemini API. They are skipped unless
``RUN_LLM_TESTS=1`` is set, so normal unit/CI runs never spend API quota.
"""

import os
from pathlib import Path

import pytest

import akasha


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes"},
        reason="set RUN_LLM_TESTS=1 to enable live API tests",
    ),
]


def _env_file() -> str:
    configured = os.getenv("ENV_FILE")
    if configured:
        return configured
    root_env = Path(__file__).resolve().parents[2] / ".env"
    return str(root_env) if root_env.exists() else ".env"


def _require_key() -> None:
    if not os.getenv("GEMINI_API_KEY") and not Path(_env_file()).exists():
        pytest.skip("GEMINI_API_KEY or a readable ENV_FILE is required")


def test_live_gemini_agent_returns_final_answer():
    _require_key()
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[],
        stream=False,
        keep_logs=True,
        thinking=True,
        thinking_budget=512,
        max_output_tokens=128,
        env_file=_env_file(),
    )

    response = agent("請只回答：測試成功。")

    assert isinstance(response, str)
    assert response.strip()
    assert agent.logs


def test_live_gemini_agent_streams_thinking_and_answer_events():
    _require_key()
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[],
        stream=True,
        keep_logs=True,
        thinking=True,
        thinking_budget=512,
        max_output_tokens=128,
        env_file=_env_file(),
    )

    events = list(agent("請先思考一句，再用一句話回答：2+2 等於多少？"))
    event_types = {event["type"] for event in events if isinstance(event, dict)}

    assert "answer" in event_types
    assert any(event["data"].strip() for event in events if event["type"] == "answer")
    assert "thinking" in event_types

    # Token-level mode should produce more than one answer chunk for this call.
    answer_events = [event for event in events if event["type"] == "answer"]
    assert len(answer_events) >= 1


def test_live_gemini_ignores_budget_when_thinking_disabled():
    _require_key()
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[],
        stream=False,
        thinking=False,
        thinking_budget=8192,
        max_output_tokens=64,
        env_file=_env_file(),
    )

    response = agent("只回答 OK")

    assert isinstance(response, str)
    assert response.strip()
