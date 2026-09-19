"""Progress through the public facade and real LangChain execution."""

import asyncio
import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

import akasha
from akasha.agent.skills import Skill
from tests.support.model_limits import model_settings

pytestmark = [pytest.mark.integration, pytest.mark.contract]


class ScriptedModel(BaseChatModel):
    replies: list
    seen: list = []
    cursor: int = 0

    @property
    def _llm_type(self):
        return "scripted-progress-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens(self, text):
        return len(text)

    def _generate(self, messages, **kwargs):
        self.seen.append(messages)
        reply = self.replies[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def _stream(self, messages, **kwargs):
        reply = self._generate(messages, **kwargs).generations[0].message
        # Text arrives before the tool call: it must not leak into answer events.
        for text in [reply.content[:3], reply.content[3:]]:
            if text:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="", tool_calls=reply.tool_calls,
            additional_kwargs=reply.additional_kwargs, chunk_position="last",
        ))


def install_model(monkeypatch, replies):
    model = ScriptedModel(replies=replies)
    monkeypatch.setattr("akasha.utils.atman.handle_model", lambda *a, **k: model)
    return model


@pytest.mark.parametrize("model_id", list(model_settings()))
@pytest.mark.parametrize("with_skill", [False, True])
def test_progress_instructions_are_automatic_and_preserve_user_prompt(monkeypatch, with_skill, model_id):
    model = install_model(monkeypatch, [AIMessage(content="完成")])
    agent = akasha.agents(
        model=model_id, system_prompt="只使用繁體中文。",
        skills=[Skill(name="research", instructions="Use reliable sources.")]
        if with_skill else None,
        keep_logs=False,
    )
    assert agent("開始") == "完成"
    prompt = model.seen[0][0].content
    assert "只使用繁體中文。" in prompt
    assert "progress" in prompt.lower()
    assert "tool" in prompt.lower()
    assert agent.system_prompt == "只使用繁體中文。"


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("narration", ["我先查詢目前庫存。", ""])
def test_stream_progress_precedes_tool_and_never_enters_answer(
    monkeypatch, capsys, thinking, narration,
):
    @tool
    def check_inventory() -> str:
        """查詢目前庫存。"""
        visible = capsys.readouterr().out
        assert "[progress]" in visible
        assert "[tool]" in visible
        return "庫存不足"

    install_model(monkeypatch, [
        AIMessage(content=narration, tool_calls=[
            {"name": "check_inventory", "args": {}, "id": "inventory-1"}
        ]),
        AIMessage(content="需要等候補貨。", additional_kwargs={"reasoning_content": "reasoning data"}),
    ])
    agent = akasha.agents(model="fake:model", tools=[check_inventory],
                          stream=True, verbose=True, thinking=thinking)
    events = list(agent("查庫存", include_thinking=False))
    progress = [e["data"] for e in events if e["type"] == "progress"]
    assert progress == [narration or "準備呼叫工具 check_inventory。"]
    assert [e["type"] for e in events] == ["progress", "tool", "answer"]
    assert agent.response == "需要等候補貨。"
    assert events[-1]["data"] == agent.response
    assert "[answer]" in capsys.readouterr().out
    log = agent.logs[agent.timestamp_list[-1]]
    assert log["progress"] == progress
    assert log["response"] == agent.response
    assert len(log["tool_calls"]) == 1
    json.dumps(log, ensure_ascii=False)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
def test_non_stream_shows_progress_before_async_tool_execution(monkeypatch, capsys, thinking, async_call):
    @tool
    async def check_inventory() -> str:
        """查詢目前庫存。"""
        assert "[progress] 我先查詢目前庫存。" in capsys.readouterr().out
        return "庫存不足"

    install_model(monkeypatch, [
        AIMessage(content="我先查詢目前庫存。", tool_calls=[
            {"name": "check_inventory", "args": {}, "id": "inventory-1"}
        ]),
        AIMessage(content="需要等候補貨。"),
    ])
    agent = akasha.agents(model="fake:model", tools=[check_inventory],
                          verbose=True, thinking=thinking)
    result = asyncio.run(agent.acall("查庫存")) if async_call else agent("查庫存")
    assert result == "需要等候補貨。"
    output = capsys.readouterr().out
    assert "[progress]" not in output  # no replay after the tool completes
    assert "[answer]" in output
    assert agent.logs[agent.timestamp_list[-1]]["progress"] == ["我先查詢目前庫存。"]


@pytest.mark.parametrize("stream", [False, True])
def test_multiple_steps_quiet_mode_and_reuse_keep_final_answer_clean(monkeypatch, capsys, stream):
    @tool
    def check_inventory() -> str:
        """查詢庫存。"""
        return "庫存不足"

    @tool
    def check_delivery() -> str:
        """查詢補貨日期。"""
        return "星期五"

    install_model(monkeypatch, [
        AIMessage(content="我先查詢目前庫存。", tool_calls=[
            {"name": "check_inventory", "args": {}, "id": "i1"}
        ]),
        AIMessage(content="庫存不足，接著確認補貨日期。", tool_calls=[
            {"name": "check_delivery", "args": {}, "id": "d1"}
        ]),
        AIMessage(content="最早可於星期五交貨。"),
        AIMessage(content="不客氣。"),
    ])
    agent = akasha.agents(model="fake:model", tools=[check_inventory, check_delivery],
                          stream=stream, verbose=False, thinking=False)
    result = agent("何時交貨？")
    if stream:
        events = list(result)
        assert [e["type"] for e in events] == ["progress", "tool", "progress", "tool", "answer"]
    else:
        assert result == "最早可於星期五交貨。"
    assert agent.response == "最早可於星期五交貨。"
    assert agent.logs[agent.timestamp_list[-1]]["progress"] == [
        "我先查詢目前庫存。", "庫存不足，接著確認補貨日期。"
    ]
    result = agent("謝謝")
    if stream:
        assert list(result) == [{"type": "answer", "data": "不客氣。"}]
    assert agent.progress == []
    assert agent.logs[agent.timestamp_list[-1]]["progress"] == []
    assert capsys.readouterr().out == ""


def test_incomplete_agent_does_not_return_progress_as_final_answer(monkeypatch):
    @tool(return_direct=True)
    def check_inventory() -> str:
        """查詢庫存。"""
        return ""

    install_model(monkeypatch, [AIMessage(content="我先查詢目前庫存。", tool_calls=[
        {"name": "check_inventory", "args": {}, "id": "i1"}
    ])])
    agent = akasha.agents(model="fake:model", tools=[check_inventory])
    with pytest.raises(RuntimeError, match="no final answer"):
        agent("查庫存")


def test_history_does_not_replay_previous_progress(monkeypatch, capsys):
    install_model(monkeypatch, [AIMessage(content="不客氣。")])
    history = [
        AIMessage(content="先查庫存。", tool_calls=[
            {"name": "check_inventory", "args": {}, "id": "old-call"}
        ]),
        ToolMessage(content="有庫存", tool_call_id="old-call", name="check_inventory"),
        AIMessage(content="有庫存。"),
    ]
    agent = akasha.agents(model="fake:model", verbose=True)
    assert agent("謝謝", messages=history) == "不客氣。"
    assert agent.progress == []
    assert "[tool]" not in capsys.readouterr().out


def test_tool_internal_model_text_is_not_an_agent_answer(monkeypatch):
    helper = ScriptedModel(replies=[AIMessage(content="工具內部草稿")])

    @tool
    def lookup() -> str:
        """查詢資料。"""
        helper.invoke("整理資料")
        return "核對完成"

    install_model(monkeypatch, [
        AIMessage(content="先查資料。", tool_calls=[
            {"name": "lookup", "args": {}, "id": "lookup-1"}
        ]),
        AIMessage(content="正式答案。"),
    ])
    agent = akasha.agents(model="fake:model", tools=[lookup], stream=True)
    events = list(agent("查資料"))
    assert [e["data"] for e in events if e["type"] == "answer"] == ["正式答案。"]
    assert agent.response == "正式答案。"


def test_fragmented_tool_arguments_are_assembled_once_with_thinking(monkeypatch):
    class FragmentedModel(ScriptedModel):
        def _stream(self, messages, **kwargs):
            if self.cursor:
                yield from super()._stream(messages, **kwargs)
                return
            self.cursor += 1
            for chunk in [
                AIMessageChunk(content="", additional_kwargs={"reasoning_content": "分析請求"}),
                AIMessageChunk(content="先查商品庫存。"),
                AIMessageChunk(content="", tool_call_chunks=[
                    {"name": "lookup", "args": '{"sku":', "id": "l1", "index": 0}
                ]),
                AIMessageChunk(content="", tool_call_chunks=[
                    {"name": None, "args": '"A123"}', "id": None, "index": 0}
                ], chunk_position="last"),
            ]:
                yield ChatGenerationChunk(message=chunk)

    seen = []

    @tool
    def lookup(sku: str) -> str:
        """查詢商品庫存。"""
        seen.append(sku)
        return "3 件"

    model = FragmentedModel(replies=[AIMessage(content=""), AIMessage(content="庫存 3 件。")])
    monkeypatch.setattr("akasha.utils.atman.handle_model", lambda *a, **k: model)
    agent = akasha.agents(model="fake:model", tools=[lookup], stream=True, thinking=True)
    events = list(agent("查 A123"))
    assert [e["type"] for e in events] == ["thinking", "progress", "tool", "answer"]
    assert seen == ["A123"]
    assert agent.tool_calls == [{"name": "lookup", "args": {"sku": "A123"}, "id": "l1", "type": "tool_call"}]
    assert agent.thoughts == ["分析請求"]
    assert agent.response == "庫存 3 件。"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_answer_preserves_findings_and_result_together(monkeypatch, stream, thinking):
    @tool
    def add_numbers() -> str:
        """Add the requested numbers."""
        return "42"

    answer = "I verified the addition result: 20 + 22 = 42."
    install_model(monkeypatch, [
        AIMessage(content="", tool_calls=[
            {"name": "add_numbers", "args": {}, "id": "add-1"}
        ]),
        AIMessage(content=answer),
    ])
    agent = akasha.agents(model="fake:model", tools=[add_numbers],
                          stream=stream, thinking=thinking)
    result = agent("Add 20 and 22.")
    if stream:
        events = list(result)
        assert "".join(e["data"] for e in events if e["type"] == "answer") == answer
        assert any(e["type"] == "progress" for e in events)
    else:
        assert result == answer
    assert agent.response == answer
