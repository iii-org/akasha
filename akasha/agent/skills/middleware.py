"""LangChain middleware for static and dynamically loaded skills."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
from threading import RLock
from typing import Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ToolCallRequest,
    dynamic_prompt,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from .loader import load_skill_directory, load_skill_metadata
from .models import Skill, SkillContext
from .registry import SkillRegistry, default_registry
from .resolver import resolve_skill_tools
from .tool_registry import SkillToolContext, ToolRegistry, default_tool_registry


@dataclass(frozen=True)
class _AvailableSkill:
    reference: str
    metadata: Skill
    source: Skill | None = None
    root: Path | None = None


class SkillAgentState(AgentState, total=False):
    loaded_skills: NotRequired[list[str]]


class _PythonExecuteInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: str
    source: str
    args: list[str] = Field(default_factory=list)
    interpreter: str | None = None
    runtime: ToolRuntime


def skill_prompt_middleware(skill_context: SkillContext, base_prompt: str = ""):
    """Return Phase 1 middleware that preserves the base prompt."""

    @dynamic_prompt
    def _skill_prompt(_request: ModelRequest) -> str:
        sections = [
            part.strip()
            for part in (base_prompt, skill_context.instructions)
            if part.strip()
        ]
        return (chr(10) + chr(10)).join(sections)

    return _skill_prompt


class DynamicSkillMiddleware(AgentMiddleware[SkillAgentState, None]):
    """Expose load_skill first, then route loaded skill tools and resources."""

    state_schema = SkillAgentState
    _default_script_timeout = 120.0
    _max_script_output_bytes = 100_000

    def __init__(
        self,
        references: str | Skill | Sequence[str | Skill],
        *,
        base_prompt: str = "",
        skill_registry: SkillRegistry = default_registry,
        tool_registry: ToolRegistry = default_tool_registry,
        tool_context: SkillToolContext | None = None,
        existing_tools: Sequence[Any] = (),
        max_resource_bytes: int = 128 * 1024,
    ) -> None:
        if not isinstance(max_resource_bytes, int) or max_resource_bytes <= 0:
            raise ValueError("max_resource_bytes must be a positive integer")

        self.base_prompt = base_prompt
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry
        self.tool_context = tool_context or SkillToolContext()
        self.existing_tools = tuple(existing_tools)
        self.max_resource_bytes = max_resource_bytes
        self._available = self._load_available(references)
        self._loaded_contexts: dict[tuple[str, ...], SkillContext] = {}
        self._loaded_tools: dict[str, Any] = {}
        self._loaded_skill_tools: dict[str, tuple[str, ...]] = {}
        self._python_repls: dict[tuple[str, str], PythonAstREPLTool] = {}
        self._python_repl_lock = RLock()
        self._resource_tool = self._make_resource_tool()
        self._python_tool = self._make_python_tool()
        self.tools = [self._make_load_skill_tool()]

    @property
    def available_context(self) -> SkillContext:
        return SkillContext(tuple(item.metadata for item in self._available))

    @property
    def available_names(self) -> list[str]:
        return [item.metadata.name for item in self._available]

    def _normalize_references(
        self, references: str | Skill | Sequence[str | Skill]
    ) -> list[str | Skill]:
        if isinstance(references, (str, Skill)):
            return [references]
        if isinstance(references, Sequence) and not isinstance(
            references, (bytes, bytearray)
        ):
            return list(references)
        raise TypeError("skills must be a name, Skill, sequence, or None")

    def _load_available(
        self, references: str | Skill | Sequence[str | Skill]
    ) -> tuple[_AvailableSkill, ...]:
        result: list[_AvailableSkill] = []
        seen: set[str] = set()

        for reference in self._normalize_references(references):
            source: Skill | None = None
            root: Path | None = None
            if isinstance(reference, Skill):
                source = reference
                metadata = Skill(
                    name=reference.name,
                    description=reference.description,
                    version=reference.version,
                    metadata=reference.metadata,
                )
                ref = reference.name
            else:
                candidate = Path(reference)
                if candidate.is_dir():
                    root = candidate.resolve()
                    metadata = load_skill_metadata(root)
                    ref = reference
                elif any(
                    separator in reference for separator in ("/", chr(92))
                ) or reference.startswith("."):
                    raise FileNotFoundError(
                        f"skill directory does not exist: {reference}"
                    )
                else:
                    source = self.skill_registry.get(reference)
                    metadata = Skill(
                        name=source.name,
                        description=source.description,
                        version=source.version,
                        metadata=source.metadata,
                    )
                    ref = reference

            if metadata.name in seen:
                continue
            seen.add(metadata.name)
            result.append(_AvailableSkill(ref, metadata, source, root))

        return tuple(result)

    def _find_available(self, reference: str) -> _AvailableSkill:
        for item in self._available:
            if reference == item.reference or reference == item.metadata.name:
                return item
        raise LookupError(
            f"skill {reference!r} is not available; choose one of "
            f"{', '.join(self.available_names) or 'none'}"
        )

    def _load_full_skill(self, item: _AvailableSkill) -> Skill:
        if item.source is not None:
            return item.source
        if item.root is not None:
            return load_skill_directory(item.root)
        return self.skill_registry.get(item.reference)

    def _make_load_skill_tool(self):
        @tool(
            "load_skill",
            description=(
                "Load one available skill by name or path. "
                "Call this before using that skill's instructions or resources."
            ),
        )
        def load_skill(reference: str, runtime: ToolRuntime) -> Command:
            item = self._find_available(reference)
            loaded = list(runtime.state.get("loaded_skills", []))
            if item.reference not in loaded:
                loaded.append(item.reference)
            message = ToolMessage(
                content=(
                    f"Skill '{item.metadata.name}' loaded. Its instructions and "
                    "resources are now available."
                ),
                name="load_skill",
                tool_call_id=runtime.tool_call_id,
            )
            return Command(update={"loaded_skills": loaded, "messages": [message]})

        return load_skill

    def _make_resource_tool(self):
        @tool(
            "read_skill_resource",
            description=(
                "Read a UTF-8 text resource from a loaded skill. "
                "Use a skill name and a path relative to that skill's root."
            ),
        )
        def read_skill_resource(
            skill: str, path: str, runtime: ToolRuntime
        ) -> str:
            loaded = runtime.state.get("loaded_skills", [])
            item = self._find_available(skill)
            if item.reference not in loaded:
                raise ValueError(
                    f"skill {item.metadata.name!r} must be loaded before reading resources"
                )
            if item.root is None:
                raise ValueError(
                    f"skill {item.metadata.name!r} has no filesystem resources"
                )
            return self._read_resource(item, path)

        return read_skill_resource

    def _make_python_tool(self):
        @tool(
            "python_execute",
            description=(
                "Execute Python with automatic routing. Pass an existing relative "
                ".py path from the loaded skill to run it in a fresh subprocess. "
                "Pass Python code generated during analysis to run it in a "
                "persistent REPL whose variables survive later calls."
            ),
            args_schema=_PythonExecuteInput,
        )
        def python_execute(
            skill: str,
            source: str,
            runtime: ToolRuntime,
            args: list[str] | None = None,
            interpreter: str | None = None,
        ) -> str:
            loaded = runtime.state.get("loaded_skills", [])
            item = self._find_available(skill)
            if item.reference not in loaded:
                raise ValueError(
                    f"skill {item.metadata.name!r} must be loaded before executing Python"
                )
            if item.root is None:
                raise ValueError(
                    f"skill {item.metadata.name!r} has no filesystem runtime"
                )
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Python source must be a non-empty string")

            value = source.strip()
            if self._is_script_source(value):
                target = self._resolve_skill_file(item, value, "script")
                return self._execute_script(item, target, args, interpreter)
            if interpreter and interpreter.strip():
                raise ValueError("interpreter override is only valid for script paths")
            if args:
                raise ValueError("args are only valid for script paths")
            return self._execute_repl(item, value, runtime)

        return python_execute

    @staticmethod
    def _is_script_source(source: str) -> bool:
        return (
            "\n" not in source
            and "\r" not in source
            and Path(source).suffix.lower() == ".py"
        )

    def _execute_script(
        self,
        item: _AvailableSkill,
        target: Path,
        args: list[str] | None,
        interpreter: str | None,
    ) -> str:
        command = self._script_command(target, interpreter)
        command.extend(str(value) for value in (args or []))
        # Skill scripts run in a child process. Force Python child processes
        # to emit UTF-8 so captured stdout/stderr can be decoded consistently
        # across Windows code pages and POSIX locales.
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=str(item.root),
                capture_output=True,
                timeout=self._default_script_timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return (
                "execution: script\n"
                f"skill script timed out after {self._default_script_timeout:g}s: "
                f"{target.name}\n{self._decode_output(exc.stdout)}"
            )
        except (FileNotFoundError, OSError) as exc:
            return (
                "execution: script\n"
                "skill script could not be started with interpreter "
                f"{command[0]!r}: {target.name}: {exc}"
            )

        stdout = self._limit_script_output(completed.stdout)
        stderr = self._limit_script_output(completed.stderr)
        result = [
            "execution: script",
            f"exit_code: {completed.returncode}",
            f"stdout:\n{stdout}",
        ]
        if stderr:
            result.append(f"stderr:\n{stderr}")
        return "\n".join(result) + "\n"
    def _execute_repl(
        self,
        item: _AvailableSkill,
        code: str,
        runtime: ToolRuntime,
    ) -> str:
        configurable = runtime.config.get("configurable", {})
        thread_id = str(configurable.get("thread_id") or "__default__")
        key = (item.reference, thread_id)
        with self._python_repl_lock:
            repl = self._python_repls.get(key)
            if repl is None:
                namespace = {"__name__": "__main__"}
                repl = PythonAstREPLTool()
                repl.globals = namespace
                repl.locals = namespace
                self._python_repls[key] = repl
            result = repl.invoke(code)
        return f"execution: repl\nstdout:\n{self._limit_script_output(result)}"

    @staticmethod
    def _script_command(target: Path, interpreter: str | None) -> list[str]:
        if interpreter and interpreter.strip():
            return [interpreter.strip(), str(target)]
        if target.suffix.lower() == ".py":
            return [sys.executable, str(target)]
        return [str(target)]

    @classmethod
    def _decode_output(cls, output: bytes | str | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output[: cls._max_script_output_bytes].decode(
                "utf-8", errors="replace"
            )
        return str(output)[: cls._max_script_output_bytes]

    @classmethod
    def _limit_script_output(cls, output: bytes | str | None) -> str:
        value = cls._decode_output(output)
        size = (
            len(output)
            if isinstance(output, bytes)
            else len(value.encode("utf-8"))
        )
        return (
            value + "\n[output truncated]"
            if size > cls._max_script_output_bytes
            else value
        )

    def _resolve_skill_file(
        self, item: _AvailableSkill, path: str, kind: str
    ) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{kind} path must be a non-empty relative path")
        relative = Path(path)
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise ValueError(f"{kind} path must stay within the skill root")
        root = item.root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{kind} path must stay within the skill root") from exc
        if not target.is_file():
            raise FileNotFoundError(f"skill {kind} is not a file: {path}")
        return target

    def _read_resource(self, item: _AvailableSkill, path: str) -> str:
        target = self._resolve_skill_file(item, path, "resource")
        if target.stat().st_size > self.max_resource_bytes:
            raise ValueError(
                f"skill resource exceeds max_resource_bytes ({self.max_resource_bytes})"
            )

        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("skill resource must be a UTF-8 text file") from exc

    def _loaded(
        self, references: Sequence[str]
    ) -> tuple[SkillContext, dict[str, Any]]:
        key = tuple(references)
        if key not in self._loaded_contexts:
            skills = tuple(
                self._load_full_skill(self._find_available(ref)) for ref in key
            )
            context = SkillContext(skills)
            resolved = resolve_skill_tools(
                context,
                self.existing_tools,
                tool_registry=self.tool_registry,
                tool_context=self.tool_context,
            )
            self._loaded_contexts[key] = context
            for tool_instance in resolved.tools:
                self._loaded_tools[tool_instance.name] = tool_instance
            self._loaded_skill_tools.update(resolved.skill_tool_names)
        return self._loaded_contexts[key], self._loaded_tools

    def _resource_available(self, references: Sequence[str]) -> bool:
        return any(self._find_available(ref).root is not None for ref in references)

    def _dynamic_tools(self, loaded_refs: Sequence[str]) -> dict[str, Any]:
        tools = dict(self._loaded_tools)
        if loaded_refs and self._resource_available(loaded_refs):
            tools[self._resource_tool.name] = self._resource_tool
            tools[self._python_tool.name] = self._python_tool
        return tools

    @property
    def loaded_skill_names(self) -> list[str]:
        return [
            skill.name
            for context in self._loaded_contexts.values()
            for skill in context.skills
        ]

    @property
    def loaded_skill_tools(self) -> dict[str, tuple[str, ...]]:
        return dict(self._loaded_skill_tools)

    def _prompt(
        self, loaded_context: SkillContext, loaded_refs: Sequence[str]
    ) -> str:
        available = [
            f"- {item.metadata.name}: {item.metadata.description} "
            f"(reference: {item.reference})"
            for item in self._available
        ]
        sections = [self.base_prompt.strip()] if self.base_prompt.strip() else []
        if available:
            sections.append(
                "Available skills:"
                + chr(10)
                + (chr(10)).join(available)
                + chr(10)
                + "Use the load_skill tool before using a skill's instructions or resources."
            )
        if loaded_context.instructions:
            sections.append(loaded_context.instructions)
        if loaded_refs and self._resource_available(loaded_refs):
            sections.append(
                "Use read_skill_resource(skill, path) to read referenced UTF-8 files "
                "under a loaded skill directory."
            )
            sections.append(
                "Use python_execute(skill, source, args) for Python work. Pass an "
                "existing relative .py path when a loaded skill references a bundled "
                "script; pass Python code generated by you for iterative calculations. "
                "Execution routing is automatic and has no mode parameter."
            )
        return (chr(10) + chr(10)).join(sections)

    def wrap_model_call(self, request: ModelRequest, handler):
        loaded_refs = request.state.get("loaded_skills", [])
        loaded_context, _ = self._loaded(loaded_refs)
        tools = list(request.tools)
        existing_names = {getattr(item, "name", None) for item in tools}
        for name, tool_instance in self._dynamic_tools(loaded_refs).items():
            if name not in existing_names:
                tools.append(tool_instance)
        return handler(
            request.override(
                tools=tools,
                system_message=SystemMessage(
                    content=self._prompt(loaded_context, loaded_refs)
                ),
            )
        )

    async def awrap_model_call(self, request: ModelRequest, handler):
        loaded_refs = request.state.get("loaded_skills", [])
        loaded_context, _ = self._loaded(loaded_refs)
        tools = list(request.tools)
        existing_names = {getattr(item, "name", None) for item in tools}
        for name, tool_instance in self._dynamic_tools(loaded_refs).items():
            if name not in existing_names:
                tools.append(tool_instance)
        return await handler(
            request.override(
                tools=tools,
                system_message=SystemMessage(
                    content=self._prompt(loaded_context, loaded_refs)
                ),
            )
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call["name"]
        dynamic_tools = self._dynamic_tools(request.state.get("loaded_skills", []))
        if name in dynamic_tools and request.tool is None:
            request = ToolCallRequest(
                tool_call=request.tool_call,
                tool=dynamic_tools[name],
                state=request.state,
                runtime=request.runtime,
            )
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call["name"]
        dynamic_tools = self._dynamic_tools(request.state.get("loaded_skills", []))
        if name in dynamic_tools and request.tool is None:
            request = ToolCallRequest(
                tool_call=request.tool_call,
                tool=dynamic_tools[name],
                state=request.state,
                runtime=request.runtime,
            )
        return await handler(request)
