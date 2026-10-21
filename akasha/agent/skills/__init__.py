"""Public skill APIs for Akasha agents."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import load_skill_directory, load_skill_metadata
    from .middleware import DynamicSkillMiddleware, SkillAgentState, skill_prompt_middleware
    from .models import Skill, SkillContext
    from .registry import SkillRegistry, default_registry
    from .resolver import ResolvedSkillTools, resolve_skill_tools, resolve_skills
    from .tool_registry import SkillToolContext, ToolRegistry, default_tool_registry

__all__ = [
    "Skill",
    "SkillContext",
    "SkillRegistry",
    "default_registry",
    "load_skill_directory",
    "load_skill_metadata",
    "resolve_skills",
    "ResolvedSkillTools",
    "resolve_skill_tools",
    "SkillToolContext",
    "ToolRegistry",
    "default_tool_registry",
    "skill_prompt_middleware",
    "DynamicSkillMiddleware",
    "SkillAgentState",
]

_LAZY_IMPORTS = {
    "Skill": (".models", "Skill"),
    "SkillContext": (".models", "SkillContext"),
    "SkillRegistry": (".registry", "SkillRegistry"),
    "default_registry": (".registry", "default_registry"),
    "load_skill_directory": (".loader", "load_skill_directory"),
    "load_skill_metadata": (".loader", "load_skill_metadata"),
    "resolve_skills": (".resolver", "resolve_skills"),
    "ResolvedSkillTools": (".resolver", "ResolvedSkillTools"),
    "resolve_skill_tools": (".resolver", "resolve_skill_tools"),
    "SkillToolContext": (".tool_registry", "SkillToolContext"),
    "ToolRegistry": (".tool_registry", "ToolRegistry"),
    "default_tool_registry": (".tool_registry", "default_tool_registry"),
    "skill_prompt_middleware": (".middleware", "skill_prompt_middleware"),
    "DynamicSkillMiddleware": (".middleware", "DynamicSkillMiddleware"),
    "SkillAgentState": (".middleware", "SkillAgentState"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
