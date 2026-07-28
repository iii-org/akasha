"""Explicit registry for skills available to Akasha agents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .loader import load_skill_directory
from .models import Skill


class SkillRegistry:
    """An allowlisted, in-process skill registry.

    A registry is intentionally explicit: a name passed to ``agents`` cannot
    cause arbitrary filesystem code to execute or import a Python module.
    """

    def __init__(self, skills: Iterable[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for skill in skills or ():
            self.register(skill)

    def register(self, skill: Skill) -> Skill:
        if skill.name in self._skills:
            raise ValueError(f"skill is already registered: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def register_directory(self, path: str | Path) -> Skill:
        return self.register(load_skill_directory(path))

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills)) or "none"
            raise LookupError(
                f"unknown skill {name!r}; registered skills: {available}"
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())


default_registry = SkillRegistry()

