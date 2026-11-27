import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

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


@pytest.mark.parametrize("include_instructions, enable_tools", [(True, True), (False, False)])
def test_preloaded_skill_is_available_on_first_model_turn(tmp_path, monkeypatch, include_instructions, enable_tools):
    from akasha import agents

    class RecordingModel(FakeChatModel):
        offered: list = Field(default_factory=list)
        prompts: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            self.offered.append({tool.name for tool in tools})
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.prompts.append(messages)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    skill = tmp_path / "research"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: research\ndescription: Research fixture.\n---\nUse reliable sources.",
        encoding="utf-8",
    )
    model = RecordingModel(chunks=[])
    monkeypatch.setattr("akasha.utils.atman.handle_model", lambda *args, **kwargs: model)
    agent = agents(model="fake:model", skills=[str(skill)], preload_skills=True,
                   include_skill_instructions=include_instructions, enable_skill_tools=enable_tools)
    assert agent("Question") == "ok"
    assert len(model.prompts) == 1
    prompt = "\n".join(message.content for message in model.prompts[0])
    assert ("Use reliable sources." in prompt) is include_instructions
    assert "Use the load_skill tool" not in prompt
    assert model.offered == ([{"read_skill_resource", "python_execute"}] if enable_tools else [])
    log = agent.logs[agent.timestamp_list[0]]
    assert log["preloaded_skills"] == ["research"]
    assert log["loaded_skills"] == []
