"""Public Agent exports without loading unrelated runtime implementations."""

from importlib import import_module
from typing import TYPE_CHECKING

# The facade shares its name with a child module; keep its class export stable.
from .agents import agents

if TYPE_CHECKING:
    from .base import create_tool
    from .mcp import normalize_mcp_result, normalize_mcp_tools
    from .skills import Skill, SkillContext, SkillRegistry, default_registry

__all__ = [
    "agents",
    "create_tool",
    "normalize_mcp_result",
    "normalize_mcp_tools",
    "Skill",
    "SkillContext",
    "SkillRegistry",
    "default_registry",
]

_LAZY_IMPORTS = {
    "create_tool": (".base", "create_tool"),
    "normalize_mcp_result": (".mcp", "normalize_mcp_result"),
    "normalize_mcp_tools": (".mcp", "normalize_mcp_tools"),
    "Skill": (".skills", "Skill"),
    "SkillContext": (".skills", "SkillContext"),
    "SkillRegistry": (".skills", "SkillRegistry"),
    "default_registry": (".skills", "default_registry"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
