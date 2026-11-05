"""Public agents use official structured chat for completion-only Ollama models."""

import asyncio
import json

import pytest
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

import akasha

pytestmark = pytest.mark.integration


@pytest.fixture
def ollama_transport(monkeypatch):
    requests = []
    replies = []

    def respond(messages, kwargs, reasoning):
        requests.append((messages, kwargs))
        assert not kwargs.get("tools"), "Ollama must not receive native tools or a failed probe"
        assert kwargs.get("format") == "json"
        reply = replies.pop(0)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        chunks = [{"message": {"content": content[i:i+12]}, "done": False}
                for i in range(0, len(content), 12)] + [
                    {"message": {"content": ""}, "done": True,
                     "prompt_eval_count": 11, "eval_count": 7}]
        if reasoning:
            chunks[0]["message"]["thinking"] = "internal reasoning"
        return chunks

    def sync(self, messages, stop=None, **kwargs):
        yield from respond(messages, kwargs, self.reasoning)

    async def async_(self, messages, stop=None, **kwargs):
        for response in respond(messages, kwargs, self.reasoning):
            yield response

    monkeypatch.setattr(ChatOllama, "_create_chat_stream", sync)
    monkeypatch.setattr(ChatOllama, "_acreate_chat_stream", async_)
    return requests, replies


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_public_agent_executes_generated_tool_and_reads_result(ollama_transport, mode):
    requests, replies = ollama_transport
    called = []

    @tool
    def read_record(record_id: int) -> str:
        """Read the current verification code for a record."""
        called.append(record_id)
        return "VERIFIED_7391"

    replies.extend([
        {"action": "read_record", "action_input": {"record_id": 42}},
        {"action": "Final Answer", "action_input": "VERIFIED_7391"},
    ])
    agent = akasha.agents(model="ollama:test-model", tools=[read_record],
                          stream=mode == "stream", max_round=3, keep_logs=True)
    if mode == "async":
        result = asyncio.run(agent.acall("Read record 42 using the tool."))
    else:
        result = agent("Read record 42 using the tool.")
    if mode == "stream":
        events = list(result)
        assert [e["type"] for e in events] == ["progress", "tool", "answer"]
        result = "".join(e["data"] for e in events if e["type"] == "answer")
        assert '"action"' not in result
    assert result == "VERIFIED_7391"
    assert called == [42]
    assert len(requests) == 2
    assert "read_record" in requests[0][0][0].content
    assert "record_id" in requests[0][0][0].content
    assert "VERIFIED_7391" in requests[1][0][-1].content
    assert agent.tool_calls[0]["args"] == {"record_id": 42}
    json.dumps(agent.logs)


@pytest.mark.parametrize("stream", [False, True])
def test_dynamic_skills_keep_the_modern_runtime_and_hidden_parameters(ollama_transport, stream):
    from akasha.agent.skills import Skill

    requests, replies = ollama_transport

    @tool
    def skill_result(value: int) -> str:
        """Return the skill's verified result."""
        return f"SKILL_{value}"

    replies.extend([
        {"action": "load_skill", "action_input": {"reference": "research"}},
        {"action": "skill_result", "action_input": {"value": 42}},
        {"action": "Final Answer", "action_input": "SKILL_42"},
    ])
    agent = akasha.agents(
        model="ollama:test-model", skills=[Skill(
            name="research", instructions="Research instruction loaded on demand.", tools=(skill_result,),
        )], stream=stream, keep_logs=True,
    )
    result = agent("Load research and return its result for 42.")
    if stream:
        result = "".join(e["data"] for e in result if e["type"] == "answer")
    assert result == "SKILL_42"
    assert "Research instruction loaded on demand." not in requests[0][0][0].content
    assert "Research instruction loaded on demand." in requests[1][0][0].content
    assert "skill_result" not in requests[0][0][0].content
    assert "skill_result" in requests[1][0][0].content
    assert "ToolRuntime" not in requests[0][0][0].content
    assert "'runtime'" not in requests[0][0][0].content
    assert agent.skill_tool_names["research"] == ("skill_result",)
    assert len(agent.logs[agent.timestamp_list[-1]]["loaded_skills"]) == 1


def test_async_only_tools_and_validation_use_the_existing_tool_runtime(ollama_transport):
    requests, replies = ollama_transport
    called = []

    @tool
    async def read_record(record_id: int) -> str:
        """Read a record by its integer ID."""
        called.append(record_id)
        return "ASYNC_VERIFIED"

    replies.extend([
        {"action": "read_record", "action_input": {"record_id": "invalid"}},
        {"action": "read_record", "action_input": {"record_id": 42}},
        {"action": "Final Answer", "action_input": "ASYNC_VERIFIED"},
    ])
    agent = akasha.agents(model="ollama:test-model", tools=[read_record], keep_logs=False)
    assert asyncio.run(agent.acall("Read record 42.")) == "ASYNC_VERIFIED"
    assert called == [42]
    assert "invalid" in requests[1][0][-1].content
    agent.stream = True
    with pytest.raises(ValueError, match="async-only"):
        agent("Read record 42.")


def test_round_limit_stops_repeated_tool_calls(ollama_transport):
    from langgraph.errors import GraphRecursionError

    requests, replies = ollama_transport

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    replies.extend([{"action": "ping", "action_input": {}}] * 8)
    agent = akasha.agents(model="ollama:test-model", tools=[ping], max_round=1, keep_logs=False)
    with pytest.raises(GraphRecursionError):
        agent("Ping forever.")
    assert len(requests) < 8


def test_history_and_system_prompt_survive_the_json_bridge(ollama_transport):
    requests, replies = ollama_transport

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    replies.append({"action": "Final Answer", "action_input": "Alice"})
    agent = akasha.agents(model="ollama:test-model", tools=[ping],
                          system_prompt="CUSTOM_SYSTEM_RULE", keep_logs=False)
    assert agent("What is my name?", messages=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice."},
    ]) == "Alice"
    assert "CUSTOM_SYSTEM_RULE" in requests[0][0][0].content
    assert requests[0][0][1].content == "My name is Alice."
    assert requests[0][0][2].content == "Hello Alice."


def test_thinking_and_usage_survive_without_exposing_protocol_json(ollama_transport):
    requests, replies = ollama_transport

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    replies.extend([
        {"action": "ping", "action_input": {}},
        {"action": "Final Answer", "action_input": "pong"},
    ])
    agent = akasha.agents(model="ollama:test-model", tools=[ping],
                          stream=True, thinking=True, keep_logs=True)
    events = list(agent("Ping."))
    assert [e["data"] for e in events if e["type"] == "thinking"] == [
        "internal reasoning", "internal reasoning",
    ]
    assert "".join(e["data"] for e in events if e["type"] == "answer") == "pong"
    assert agent.messages[-1]["usage_metadata"]["total_tokens"] == 18
    assert len(requests) == 2


def test_invalid_json_propagates_without_executing_or_retrying(ollama_transport):
    from langchain_core.exceptions import OutputParserException

    requests, replies = ollama_transport
    called = []

    @tool
    def ping() -> str:
        """Return pong."""
        called.append(True)
        return "pong"

    replies.append("not JSON")
    agent = akasha.agents(model="ollama:test-model", tools=[ping], keep_logs=False)
    with pytest.raises(OutputParserException):
        agent("Ping.")
    assert called == []
    assert len(requests) == 1
