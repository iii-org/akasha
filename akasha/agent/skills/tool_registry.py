"""Allowlisted factories for skill-provided tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Iterable

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class SkillToolContext:
    """Safe, explicit configuration exposed to a skill tool factory."""

    env_file: str = ""
    language: str = "ch"
    model: str = ""


ToolFactory = Callable[..., BaseTool]


class ToolRegistry:
    """An explicit allowlist of factories available to skill definitions."""

    def __init__(self, tools: Iterable[tuple[str, ToolFactory]] | None = None):
        self._factories: dict[str, ToolFactory] = {}
        for name, factory in tools or ():
            self.register(name, factory)

    def register(self, name: str, factory: ToolFactory) -> ToolFactory:
        if not name or not name.strip():
            raise ValueError("tool registry name cannot be empty")
        if not callable(factory):
            raise TypeError("tool factory must be callable")
        if name in self._factories:
            raise ValueError(f"tool is already registered: {name}")
        self._factories[name] = factory
        return factory

    def create(self, name: str, context: SkillToolContext) -> BaseTool:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "none"
            raise LookupError(
                f"unknown skill tool {name!r}; registered tools: {available}"
            ) from exc

        parameters = inspect.signature(factory).parameters
        tool = factory() if not parameters else factory(context)
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"skill tool factory {name!r} must return BaseTool, "
                f"got {type(tool).__name__}"
            )
        if tool.name != name:
            raise ValueError(
                f"skill tool registry name {name!r} does not match tool name "
                f"{tool.name!r}"
            )
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._factories


default_tool_registry = ToolRegistry()

