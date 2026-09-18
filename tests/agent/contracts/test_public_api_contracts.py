"""Public API contracts that do not require RAG, MCP, or provider credentials."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from akasha.agent.base import create_tool
from akasha.helper.run_llm import call_stream_events
from tests.support.fakes import FakeChatModel

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_ask_stream_events_have_stable_json_contract(capsys):
    model = FakeChatModel(
        chunks=[
            AIMessageChunk(
                content=[
                    {"type": "thinking", "thinking": "private plan"},
                    {"type": "text", "text": "final answer"},
                ]
            )
        ]
    )

    events = list(
        call_stream_events(
            model,
            "question",
            include_thinking=True,
            verbose=False,
        )
    )

    assert events == [
        {"type": "thinking", "data": "private plan"},
        {"type": "answer", "data": "final answer"},
    ]
    assert all(set(event) == {"type", "data"} for event in events)
    json.dumps(events, ensure_ascii=False)
    assert capsys.readouterr().out == ""


def test_agents_non_stream_returns_string_and_serializable_logs(monkeypatch):
    agents_module = __import__("akasha.agent.agents", fromlist=["agents"])

    class FakeAgent:
        async def ainvoke(self, _payload, config=None):
            return {"messages": [AIMessage(content="final answer")]}

    monkeypatch.setattr(
        "akasha.utils.atman.handle_model",
        lambda *args, **kwargs: FakeChatModel(chunks=[]),
    )
    monkeypatch.setattr(agents_module, "create_agent", lambda **kwargs: FakeAgent())

    agent = agents_module.agents(model="fake:model", keep_logs=True)
    result = asyncio.run(agent.acall("question"))

    assert result == "final answer"
    assert agent.response == "final answer"
    assert len(agent.timestamp_list) == 1
    json.dumps(agent.logs, ensure_ascii=False)
    log = agent.logs[agent.timestamp_list[0]]
    assert log["response"] == "final answer"
    assert log["messages"]


def test_agents_stream_normalizes_tool_thinking_and_answer_events(monkeypatch):
    agents_module = __import__("akasha.agent.agents", fromlist=["agents"])
    calls = []

    def add(left: int, right: int) -> int:
        calls.append((left, right))
        return left + right

    add_tool = create_tool("Add two numbers.", add, "add")

    class FakeAgent:
        def stream(self, _payload, config=None, stream_mode=None):
            assert stream_mode == "messages"
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"left": 2, "right": 3}, "id": "call-1"}
                ],
            )
            assert add_tool.invoke({"left": 2, "right": 3}) == 5
            yield ToolMessage(content="5", tool_call_id="call-1", name="add")
            yield AIMessageChunk(
                content=[
                    {"type": "thinking", "thinking": "use tool result"},
                    {"type": "text", "text": "5"},
                ]
            )

    monkeypatch.setattr(
        "akasha.utils.atman.handle_model",
        lambda *args, **kwargs: FakeChatModel(chunks=[]),
    )
    monkeypatch.setattr(agents_module, "create_agent", lambda **kwargs: FakeAgent())

    agent = agents_module.agents(
        model="fake:model",
        tools=[add_tool],
        stream=True,
        thinking=True,
        keep_logs=True,
    )
    events = list(agent("calculate 2 + 3"))

    assert [event["type"] for event in events] == ["progress", "tool", "thinking", "answer"]
    assert events[-2:] == [
        {"type": "thinking", "data": "use tool result"},
        {"type": "answer", "data": "5"},
    ]
    assert calls == [(2, 3)]
    json.dumps(events, ensure_ascii=False)
    json.dumps(agent.logs, ensure_ascii=False)
    assert agent.logs[agent.timestamp_list[0]]["response"] == "5"
