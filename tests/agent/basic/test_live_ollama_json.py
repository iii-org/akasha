"""Opt-in acceptance through akasha.agents against a real Ollama deployment."""

import asyncio
from datetime import datetime, timedelta, timezone
import os
import random
import secrets

import pytest

import akasha

pytestmark = [pytest.mark.live, pytest.mark.requires_api]


def _model():
    model = os.getenv("AKASHA_OLLAMA_AGENT_MODEL")
    if not model:
        pytest.skip("set AKASHA_OLLAMA_AGENT_MODEL to an ollama: model specification")
    return model


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_real_agent_reads_unpredictable_result(mode, monkeypatch):
    from langchain_classic.agents.output_parsers.json import JSONAgentOutputParser

    decisions = []
    parse = JSONAgentOutputParser.parse

    def capture(self, text):
        decisions.append(text)
        return parse(self, text)

    monkeypatch.setattr(JSONAgentOutputParser, "parse", capture)
    model = _model()
    code = "VERIFIED_" + secrets.token_hex(8)
    calls = []

    def lookup_code(record_id: int) -> str:
        """查詢指定record_id的驗證碼。驗證碼必須呼叫工具才能取得，不能猜測。"""
        calls.append(record_id)
        return code

    agent = akasha.agents(
        model=model, tools=[akasha.create_tool(lookup_code.__doc__, lookup_code, "lookup_code")],
        stream=mode == "stream", max_output_tokens=512, max_round=4, keep_logs=True,
    )
    question = "請呼叫lookup_code查詢record_id=42的驗證碼，最後原樣回傳驗證碼。"
    result = asyncio.run(agent.acall(question)) if mode == "async" else agent(question)
    if mode == "stream":
        events = list(result)
        assert any(e["type"] == "tool" for e in events)
        result = "".join(e["data"] for e in events if e["type"] == "answer")
        assert '"action"' not in result
    assert calls == [42], {"actual_calls": calls, "answer": result, "tool_calls": agent.tool_calls}
    assert code in result, {"answer": result, "raw_final_decision": decisions[-1]}
    assert agent.tool_calls[0]["args"] == {"record_id": 42}


@pytest.mark.parametrize("stream", [False, True])
def test_real_agent_chains_the_users_three_tools(stream):
    model = _model()
    observed = []

    def get_current_datetime() -> str:
        """取得台灣目前日期與時間。"""
        result = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")
        observed.append(("get_current_datetime", {}, result))
        return result

    def add_numbers(a: int, b: int) -> int:
        """將兩個整數相加。"""
        observed.append(("add_numbers", {"a": a, "b": b}, a+b))
        return a+b

    def shuffle_numbers(numbers: list[int]) -> list[int]:
        """隨機排列整數列表，不修改原始列表。

        若要打散一個數值的各位數，先將數值轉成數字列表，例如123轉成[1,2,3]。
        """
        result = numbers[:]
        random.shuffle(result)
        observed.append(("shuffle_numbers", {"numbers": numbers}, result))
        return result

    tools = [akasha.create_tool(f.__doc__, f, f.__name__)
             for f in (get_current_datetime, add_numbers, shuffle_numbers)]
    agent = akasha.agents(model=model, tools=tools, stream=stream,
                          max_output_tokens=1024, max_round=8, keep_logs=True)
    result = agent(
        "取得現在的日期時間，將年份加上77，再將相加結果的各位數隨機排列，"
        "告訴我過程與最後結果，最後原樣引用shuffle_numbers回傳的列表。"
    )
    if stream:
        events = list(result)
        assert len([e for e in events if e["type"] == "tool"]) == 3
        result = "".join(e["data"] for e in events if e["type"] == "answer")
    assert [o[0] for o in observed] == [f.name for f in tools]
    year = int(observed[0][2][:4])
    assert sorted(observed[1][1].values()) == sorted([year, 77])
    assert observed[2][1]["numbers"] == [int(c) for c in str(year+77)]
    assert str(observed[2][2]).replace(" ", "") in result.replace(" ", "")
    assert str(year+77) in result
    assert len(agent.tool_calls) == 3


@pytest.mark.parametrize("stream", [False, True])
def test_real_plain_chat_and_agent_without_tools(stream):
    model = _model()
    question = "請只回覆OLLAMA_CHAT_OK，不要其他文字。"
    qa = akasha.ask(model=model, stream=stream, keep_logs=False, max_output_tokens=128)
    result = qa(question)
    if stream:
        result = "".join(result)
    assert "OLLAMA_CHAT_OK" in result
    agent = akasha.agents(model=model, stream=stream, keep_logs=False, max_output_tokens=128)
    result = agent(question)
    if stream:
        result = "".join(e["data"] for e in result if e["type"] == "answer")
    assert "OLLAMA_CHAT_OK" in result
    assert not agent.tool_calls


def test_real_agent_loads_skill_and_executes_its_tool():
    from akasha.agent.skills import Skill

    model = _model()
    code = "SKILL_" + secrets.token_hex(8)
    calls = []

    def skill_code(record_id: int) -> str:
        """查詢指定record_id的驗證碼，必須呼叫本工具才能取得。"""
        calls.append(record_id)
        return code

    tool = akasha.create_tool(skill_code.__doc__, skill_code, "skill_code")
    skill = Skill(name="verification", description="提供驗證碼查詢工具。",
                  instructions="請用skill_code工具查詢驗證碼，忠實回傳其結果。", tools=(tool,))
    agent = akasha.agents(model=model, skills=[skill], max_round=6,
                          max_output_tokens=512, keep_logs=True)
    result = agent("先載入verification Skill，再呼叫其skill_code查詢record_id=42，原樣回傳驗證碼。")
    assert calls == [42]
    assert code in result
    assert [c["name"] for c in agent.tool_calls] == ["load_skill", "skill_code"]
    assert agent.skill_tool_names["verification"] == ("skill_code",)


def test_real_agent_can_be_called_twice():
    model = _model()
    observed = []

    def lookup_code() -> str:
        """取得本次最新驗證碼，每次查詢都必須重新呼叫本工具。"""
        code = "REUSE_" + secrets.token_hex(8)
        observed.append(code)
        return code

    agent = akasha.agents(
        model=model, tools=[akasha.create_tool(lookup_code.__doc__, lookup_code, "lookup_code")],
        max_round=4, max_output_tokens=512, keep_logs=False,
    )
    for i in range(2):
        result = agent("請呼叫lookup_code取得本次最新驗證碼，最後原樣回傳該驗證碼。")
        assert len(observed) == i+1
        assert observed[-1] in result
        assert len(agent.tool_calls) == 1
