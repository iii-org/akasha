"""Opt-in live tests for the LangChain-native Gemini Agent path.

These tests intentionally call the real Gemini API. They are skipped unless
``RUN_LIVE_TESTS=1`` is set, so normal local runs never spend API quota.
"""

import pytest

import akasha
from tests.support.model_limits import output_token_budget
from tests.support.live import load_test_env, require_keys


pytestmark = [
    pytest.mark.live,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


def _env_file() -> str:
    return load_test_env()


def _require_key() -> None:
    require_keys("GEMINI_API_KEY")


def test_live_gemini_agent_returns_final_answer():
    _require_key()
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[],
        stream=False,
        keep_logs=True,
        thinking=True,
        thinking_budget=512,
        max_output_tokens=output_token_budget("gemini:gemini-2.5-flash"),
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
        max_output_tokens=output_token_budget("gemini:gemini-2.5-flash"),
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
        max_output_tokens=output_token_budget("gemini:gemini-2.5-flash"),
        env_file=_env_file(),
    )

    response = agent("只回答 OK")

    assert isinstance(response, str)
    assert response.strip()
