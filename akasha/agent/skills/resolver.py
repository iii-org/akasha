"""Normalize skills and resolve their allowlisted tool bundles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from .loader import load_skill_directory
from .models import Skill, SkillContext
from .registry import SkillRegistry, default_registry
from .tool_registry import SkillToolContext, ToolRegistry, default_tool_registry


def resolve_skills(
    skills: str | Skill | Sequence[str | Skill] | None,
    registry: SkillRegistry = default_registry,
) -> SkillContext:
    if skills is None:
        return SkillContext()
    if isinstance(skills, (str, Skill)):
        values: list[Any] = [skills]
    elif isinstance(skills, Sequence) and not isinstance(skills, (bytes, bytearray)):
        values = list(skills)
    else:
        raise TypeError("skills must be a name, Skill, sequence, or None")

    resolved: list[Skill] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            candidate = Path(value)
            if candidate.is_dir():
                skill = load_skill_directory(candidate)
            elif any(separator in value for separator in ("/", "\\\\")) or value.startswith("."):
                raise FileNotFoundError(f"skill directory does not exist: {value}")
            else:
                skill = registry.get(value)
        else:
            skill = value
        if not isinstance(skill, Skill):
            raise TypeError("skills must contain only names or Skill instances")
        if skill.name in seen:
            continue
        seen.add(skill.name)
        resolved.append(skill)
    return SkillContext(tuple(resolved))


@dataclass(frozen=True)
class ResolvedSkillTools:
    """Tools created for one agent, grouped by their source skill."""

    tools: tuple[BaseTool, ...] = ()
    skill_tool_names: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "skill_tool_names",
            dict(self.skill_tool_names or {}),
        )


def resolve_skill_tools(
    skill_context: SkillContext,
    existing_tools: Sequence[BaseTool],
    tool_registry: ToolRegistry = default_tool_registry,
    tool_context: SkillToolContext | None = None,
) -> ResolvedSkillTools:
    """Create and validate skill tools without replacing existing tools."""

    context = tool_context or SkillToolContext()
    existing_names: set[str] = set()
    for tool in existing_tools:
        if not isinstance(tool, BaseTool):
            raise TypeError("existing tools must contain only BaseTool instances")
        if tool.name in existing_names:
            raise ValueError(f"duplicate agent tool name: {tool.name}")
        existing_names.add(tool.name)

    resolved: list[BaseTool] = []
    resolved_names: set[str] = set(existing_names)
    grouped: dict[str, tuple[str, ...]] = {}

    for skill in skill_context.skills:
        names: list[str] = []
        seen_in_skill: set[str] = set()
        candidates = list(skill.tools)
        for name in skill.tool_names:
            if name in seen_in_skill:
                continue
            seen_in_skill.add(name)
            candidates.append(tool_registry.create(name, context))

        for tool in candidates:
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    f"skill {skill.name!r} contains a non-BaseTool instance"
                )
            if tool.name in resolved_names:
                raise ValueError(
                    f"skill tool name {tool.name!r} conflicts with an existing "
                    "agent or skill tool"
                )
            resolved_names.add(tool.name)
            resolved.append(tool)
            names.append(tool.name)
        grouped[skill.name] = tuple(names)

    return ResolvedSkillTools(tuple(resolved), grouped)
