from pathlib import Path
import sys

import pytest
from langchain.tools import ToolRuntime

from akasha.agent.base import create_tool
from akasha.agent.skills import (
    DynamicSkillMiddleware,
    Skill,
    SkillContext,
    SkillToolContext,
    ToolRegistry,
    resolve_skill_tools,
)


pytestmark = pytest.mark.unit


def _tool(value: str):
    def read_value() -> str:
        return value

    return create_tool("Return a fixed value.", read_value, value)


def test_resolve_skill_tool_names_uses_factory_context():
    seen = []
    registry = ToolRegistry()

    def factory(context: SkillToolContext):
        seen.append(context)
        return _tool("context-aware")

    registry.register("context-aware", factory)
    context = SkillToolContext(env_file="test.env", language="en", model="fake:model")
    skill_context = SkillContext(
        (Skill(name="research", tool_names=("context-aware",)),)
    )

    resolved = resolve_skill_tools(skill_context, [], registry, context)

    assert [tool.name for tool in resolved.tools] == ["context-aware"]
    assert resolved.skill_tool_names == {"research": ("context-aware",)}
    assert seen == [context]


def test_resolve_skill_tools_rejects_collision_with_existing_tool():
    tool = _tool("search")
    registry = ToolRegistry()
    registry.register("search", lambda: tool)
    skill_context = SkillContext((Skill(name="research", tool_names=("search",)),))

    with pytest.raises(ValueError, match="conflicts"):
        resolve_skill_tools(skill_context, [tool], registry)


def test_resolve_skill_tools_accepts_trusted_direct_tool_instances():
    tool = _tool("direct")
    skill_context = SkillContext((Skill(name="test", tools=(tool,)),))

    resolved = resolve_skill_tools(skill_context, [])

    assert resolved.tools == (tool,)
    assert resolved.skill_tool_names == {"test": ("direct",)}


def test_dynamic_skill_loads_instructions_and_tools_after_load():
    registry = ToolRegistry()
    seen = []

    def factory(context: SkillToolContext):
        seen.append(context)

        def loaded() -> str:
            return "loaded"

        return create_tool("Return a fixed value.", loaded, "context-aware")

    registry.register("context-aware", factory)
    skill = Skill(
        name="research",
        description="Research",
        instructions="Use the research workflow.",
        tool_names=("context-aware",),
    )
    middleware = DynamicSkillMiddleware([skill], tool_registry=registry)

    assert middleware.available_context.skills[0].instructions == ""
    assert seen == []

    runtime = ToolRuntime(
        state={},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="load-1",
        store=None,
    )
    update = middleware.tools[0].invoke(
        {"reference": "research", "runtime": runtime}
    )

    assert update.update["loaded_skills"] == ["research"]

    context, tools = middleware._loaded(["research"])
    assert context.skills[0].instructions == skill.instructions
    assert [tool.name for tool in tools.values()] == ["context-aware"]
    assert seen == [middleware.tool_context]


def test_filesystem_skill_resources_are_lazy_and_safe():
    skill_dir = Path(__file__).parents[1] / "fixtures" / "skills" / "research"
    middleware = DynamicSkillMiddleware([str(skill_dir)])

    assert [item.name for item in middleware.tools] == ["load_skill"]
    assert "read_skill_resource" not in middleware._dynamic_tools([])

    loaded = [str(skill_dir)]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="resource-1",
        store=None,
    )
    middleware._loaded(loaded)
    dynamic = middleware._dynamic_tools(loaded)

    assert "read_skill_resource" in dynamic
    assert dynamic["read_skill_resource"].invoke(
        {
            "skill": "research",
            "path": "references/REFERENCE.md",
            "runtime": runtime,
        }
    ).startswith("# Reference")
    assert dynamic["read_skill_resource"].invoke(
        {
            "skill": "research",
            "path": "assets/config.json",
            "runtime": runtime,
        }
    ) == '{"enabled": true}'

    script_source = dynamic["read_skill_resource"].invoke(
        {
            "skill": "research",
            "path": "scripts/read_data.py",
            "runtime": runtime,
        }
    )
    assert 'print("executed"' in script_source

    with pytest.raises(ValueError, match="within the skill root"):
        middleware._read_resource(
            middleware._find_available("research"), "../outside.txt"
        )
    with pytest.raises(ValueError, match="within the skill root"):
        middleware._read_resource(
            middleware._find_available("research"), str(skill_dir / "SKILL.md")
        )
    with pytest.raises(FileNotFoundError, match="not a file"):
        middleware._read_resource(
            middleware._find_available("research"), "references"
        )


def test_filesystem_resource_size_limit_is_configurable():
    skill_dir = Path(__file__).parents[1] / "fixtures" / "skills" / "research"
    middleware = DynamicSkillMiddleware(
        [str(skill_dir)],
        max_resource_bytes=10,
    )

    with pytest.raises(ValueError, match="max_resource_bytes"):
        middleware._read_resource(
            middleware._find_available("research"), "references/large.txt"
        )

def _temporary_script_skill(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "runtime-test"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\n"
        "name: runtime-test\n"
        "description: Runtime test skill\n"
        "---\n\n"
        "Run scripts only when a test requests them.\n",
        encoding="utf-8",
    )
    (scripts / "run.py").write_text(source, encoding="utf-8")
    return root


def _invoke_script(
    skill_dir: Path,
    *,
    args: list[str] | None = None,
    interpreter: str | None = None,
) -> tuple[DynamicSkillMiddleware, str]:
    reference = str(skill_dir)
    middleware = DynamicSkillMiddleware([reference])
    loaded = [reference]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="script-runtime-test",
        store=None,
    )
    middleware._loaded(loaded)
    script_tool = middleware._dynamic_tools(loaded)["python_execute"]
    result = script_tool.invoke(
        {
            "skill": "runtime-test",
            "source": "scripts/run.py",
            "args": args or [],
            "interpreter": interpreter,
            "runtime": runtime,
        }
    )
    return middleware, result


def test_skill_script_uses_caller_runtime_and_returns_process_result(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AKASHA_SKILL_TEST_VALUE", "inherited")
    skill_dir = _temporary_script_skill(
        tmp_path,
        "\n".join(
            [
                "import os",
                "import sys",
                "import pytest",
                "print(f'executable:{sys.executable}')",
                "print(f'env:{os.environ[\"AKASHA_SKILL_TEST_VALUE\"]}')",
                "print(f'args:{sys.argv[1:]}')",
                "print(f'caller-package:{pytest.__name__}')",
                "print('script warning', file=sys.stderr)",
                "raise SystemExit(3)",
            ]
        ),
    )

    middleware, result = _invoke_script(skill_dir, args=["alpha", "two words"])

    assert middleware._python_tool.args.keys() == {
        "skill",
        "source",
        "args",
        "interpreter",
    }
    assert "execution: script" in result
    assert "exit_code: 3" in result
    assert f"executable:{sys.executable}" in result
    assert "env:inherited" in result
    assert "args:['alpha', 'two words']" in result
    assert "caller-package:pytest" in result
    assert "stderr:\nscript warning" in result


def test_python_execute_repl_supports_normal_code_and_persists_by_thread(tmp_path):
    skill_dir = _temporary_script_skill(tmp_path, "print('unused')\n")
    reference = str(skill_dir)
    middleware = DynamicSkillMiddleware([reference])
    loaded = [reference]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={"configurable": {"thread_id": "repl-thread"}},
        stream_writer=lambda _: None,
        tool_call_id="python-repl-test",
        store=None,
    )
    middleware._loaded(loaded)
    python_tool = middleware._dynamic_tools(loaded)["python_execute"]

    first = python_tool.invoke(
        {
            "skill": "runtime-test",
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
    second = python_tool.invoke(
        {
            "skill": "runtime-test",
            "source": "calculate_average(values)",
            "runtime": runtime,
        }
    )

    assert "execution: repl" in first
    assert "20" in first
    assert "execution: repl" in second
    assert "5" in second
def test_skill_script_timeout_is_reported_without_crashing(tmp_path):
    skill_dir = _temporary_script_skill(
        tmp_path,
        "import time\nprint('started', flush=True)\ntime.sleep(1)\n",
    )
    reference = str(skill_dir)
    middleware = DynamicSkillMiddleware([reference])
    middleware._default_script_timeout = 0.05
    loaded = [reference]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="script-timeout-test",
        store=None,
    )
    middleware._loaded(loaded)

    result = middleware._dynamic_tools(loaded)["python_execute"].invoke(
        {
            "skill": "runtime-test",
            "source": "scripts/run.py",
            "runtime": runtime,
        }
    )

    assert "timed out after 0.05s" in result


def test_skill_script_output_is_truncated(tmp_path, monkeypatch):
    skill_dir = _temporary_script_skill(
        tmp_path,
        "import sys\nprint('x' * 200)\nprint('y' * 200, file=sys.stderr)\n",
    )
    reference = str(skill_dir)
    middleware = DynamicSkillMiddleware([reference])
    monkeypatch.setattr(DynamicSkillMiddleware, "_max_script_output_bytes", 32)
    loaded = [reference]
    runtime = ToolRuntime(
        state={"loaded_skills": loaded},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="script-output-test",
        store=None,
    )
    middleware._loaded(loaded)

    result = middleware._dynamic_tools(loaded)["python_execute"].invoke(
        {
            "skill": "runtime-test",
            "source": "scripts/run.py",
            "runtime": runtime,
        }
    )

    assert result.count("[output truncated]") == 2
    assert "exit_code: 0" in result


def test_skill_script_missing_interpreter_and_module_are_reported(tmp_path):
    skill_dir = _temporary_script_skill(
        tmp_path,
        "import akasha_module_that_does_not_exist\n",
    )

    _, interpreter_result = _invoke_script(
        skill_dir,
        interpreter="akasha-interpreter-that-does-not-exist",
    )
    assert "could not be started" in interpreter_result
    assert "akasha-interpreter-that-does-not-exist" in interpreter_result

    _, module_result = _invoke_script(skill_dir)
    assert "exit_code: 1" in module_result
    assert "ModuleNotFoundError" in module_result
    assert "akasha_module_that_does_not_exist" in module_result


def test_skill_script_rejects_absolute_directory_and_symlink_paths(tmp_path):
    skill_dir = _temporary_script_skill(tmp_path, "print('inside')\n")
    middleware = DynamicSkillMiddleware([str(skill_dir)])
    item = middleware._find_available("runtime-test")
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="within the skill root"):
        middleware._resolve_skill_file(item, str(outside.resolve()), "script")
    with pytest.raises(FileNotFoundError, match="not a file"):
        middleware._resolve_skill_file(item, "scripts", "script")

    link = skill_dir / "scripts" / "outside-link.py"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="within the skill root"):
        middleware._resolve_skill_file(item, "scripts/outside-link.py", "script")


def test_loading_skill_does_not_execute_skill_yaml_or_install_dependencies(
    tmp_path,
):
    skill_dir = _temporary_script_skill(
        tmp_path,
        "import akasha_dependency_that_does_not_exist\n",
    )
    marker = tmp_path / "skill-yaml-executed"
    (skill_dir / "skill.yaml").write_text(
        "script: scripts/run.py\n"
        f"marker: {marker}\n",
        encoding="utf-8",
    )

    middleware = DynamicSkillMiddleware([str(skill_dir)])
    middleware._loaded([str(skill_dir)])

    assert not marker.exists()
    _, result = _invoke_script(skill_dir)
    assert "ModuleNotFoundError" in result
    assert not marker.exists()
