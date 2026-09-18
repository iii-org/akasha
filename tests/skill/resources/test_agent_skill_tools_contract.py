import pytest

from akasha.agent.base import create_tool
from akasha.agent.skills import Skill, default_tool_registry
from tests.support.fakes import FakeChatModel
from tests.support.paths import FIXTURES_ROOT


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_agent_defers_skill_tools_until_skill_is_loaded(monkeypatch):
    agents_module = __import__("akasha.agent.agents", fromlist=["agents"])
    captured = {}

    def existing() -> str:
        return "existing"

    def skill_action() -> str:
        return "skill"

    existing_tool = create_tool("Existing tool.", existing, "existing_tool")
    skill_tool = create_tool("Skill tool.", skill_action, "skill_action")
    registry_name = "skill_action"
    default_tool_registry.register(registry_name, lambda: skill_tool)

    class FakeAgent:
        pass

    monkeypatch.setattr(
        "akasha.utils.atman.handle_model",
        lambda *args, **kwargs: _fake_model(),
    )
    monkeypatch.setattr(
        agents_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or FakeAgent(),
    )

    agents_module.agents(
        model="fake:model",
        tools=[existing_tool],
        skills=[Skill(name="research", tool_names=(registry_name,))],
    )

    assert [tool.name for tool in captured["tools"]] == ["existing_tool"]
    assert [tool.name for tool in captured["middleware"][0].tools] == ["load_skill"]


def _fake_model():
    return FakeChatModel(chunks=[])

def test_loaded_filesystem_skill_exposes_automatic_script_execution():
    from pathlib import Path
    from langchain.tools import ToolRuntime
    from akasha.agent.skills import DynamicSkillMiddleware

    skill_dir = FIXTURES_ROOT / "skills" / "research"
    middleware = DynamicSkillMiddleware([str(skill_dir)])
    loaded = [str(skill_dir)]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="script-1",
        store=None,
    )

    middleware._loaded(loaded)
    dynamic = middleware._dynamic_tools(loaded)
    assert "python_execute" in dynamic
    result = dynamic["python_execute"].invoke(
        {
            "skill": "research",
            "source": "scripts/read_data.py",
            "args": ["hello"],
            "runtime": runtime,
        }
    )
    assert "exit_code: 0" in result
    assert "executed hello" in result


def test_script_runner_rejects_paths_outside_skill_root():
    from pathlib import Path
    from akasha.agent.skills import DynamicSkillMiddleware

    skill_dir = FIXTURES_ROOT / "skills" / "research"
    middleware = DynamicSkillMiddleware([str(skill_dir)])

    with pytest.raises(ValueError, match="within the skill root"):
        middleware._resolve_skill_file(
            middleware._find_available("research"),
            "../outside.py",
            "script",
        )


def test_loaded_filesystem_skill_exposes_persistent_python_repl():
    from pathlib import Path
    from langchain.tools import ToolRuntime
    from akasha.agent.skills import DynamicSkillMiddleware

    skill_dir = FIXTURES_ROOT / "skills" / "research"
    middleware = DynamicSkillMiddleware([str(skill_dir)])
    loaded = [str(skill_dir)]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="python-repl-1",
        store=None,
    )

    middleware._loaded(loaded)
    dynamic = middleware._dynamic_tools(loaded)
    assert "python_execute" in dynamic

    first = dynamic["python_execute"].invoke(
        {
            "skill": "research",
            "source": (
                "import statistics\n"
                "values = [2, 4, 6, 8]\n"
                "def calculate_average(items):\n"
                "    return statistics.mean(items)\n"
                "total = sum(values)\n"
                "total"
            ),
            "runtime": runtime,
        }
    )
    second = dynamic["python_execute"].invoke(
        {
            "skill": "research",
            "source": "calculate_average(values)",
            "runtime": runtime,
        }
    )

    assert "execution: repl" in first
    assert "20" in first
    assert "execution: repl" in second
    assert "5" in second
