import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from akasha.agent.skills import Skill
from tests.support.fakes import FakeChatModel


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_agents_keep_tools_and_add_skill_middleware(monkeypatch):
    agents_module = __import__("akasha.agent.agents", fromlist=["agents"])
    captured = {}

    class FakeAgent:
        async def ainvoke(self, _payload, config=None):
            return {"messages": [AIMessage(content="ok")]}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(
        "akasha.utils.atman.handle_model",
        lambda *args, **kwargs: _fake_model(),
    )
    monkeypatch.setattr(agents_module, "create_agent", fake_create_agent)

    skill = Skill(name="research", instructions="Use reliable sources.", version="1")
    agent = agents_module.agents(
        model="fake:model",
        tools=[],
        skills=[skill],
        system_prompt="Base instructions.",
        keep_logs=True,
    )
    result = asyncio.run(agent.acall("question"))

    assert result == "ok"
    assert captured["tools"] == []
    assert len(captured["middleware"]) == 1
    log = agent.logs[agent.timestamp_list[0]]
    assert log["skills"] == ["research"]
    assert log["skill_versions"] == {"research": "1"}
    json.dumps(agent.logs, ensure_ascii=False)


def _fake_model():
    return FakeChatModel(chunks=[])
