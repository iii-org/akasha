"""Data models for Akasha agent skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class Skill:
    """A named set of instructions and optionally registered tools.

    Phase 1 uses ``instructions`` only.  The ``tools`` field is kept in the
    model so the public abstraction can grow into tool bundles without
    changing the ``akasha.agents(..., skills=...)`` API.
    """

    name: str
    description: str = ""
    instructions: str = ""
    tools: tuple[BaseTool, ...] = ()
    tool_names: tuple[str, ...] = ()
    version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("skill name cannot be empty")
        if not isinstance(self.instructions, str):
            raise TypeError("skill instructions must be a string")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True)
class SkillContext:
    """Resolved, immutable skill information used by agent middleware."""

    skills: tuple[Skill, ...] = ()

    @property
    def instructions(self) -> str:
        sections = []
        for skill in self.skills:
            text = skill.instructions.strip()
            if text:
                sections.append(f"## Skill: {skill.name}\n{text}")
        return "\n\n".join(sections)

    @property
    def names(self) -> list[str]:
        return [skill.name for skill in self.skills]

    @property
    def versions(self) -> dict[str, str]:
        return {skill.name: skill.version for skill in self.skills if skill.version}

