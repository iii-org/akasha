from importlib import import_module
import os

# Optional automatic logging configuration.
# Enable by setting AKASHA_AUTO_CONFIGURE_LOGGING to "1", "true", or "yes".
os.environ["CHROMA_TELEMETRY_OPT_OUT"] = "True"
if os.getenv("AKASHA_AUTO_CONFIGURE_LOGGING", "").lower() in {"1", "true", "yes"}:
    from akasha.utils.logging_config import configure_logging

    configure_logging(verbose=True, keep_logs=False)

__all__ = [
    "RAG",
    "ask",
    "summary",
    "websearch",
    "eval",
    "agents",
    "create_tool",
    "Skill",
    "SkillRegistry",
    "default_registry",
    "gen_image",
    "edit_image",
    "MemoryManager",
]

_LAZY_IMPORTS = {
    "RAG": (".RAG.rag", "RAG"),
    "ask": (".tools.ask", "ask"),
    "summary": (".tools.summary", "summary"),
    "websearch": (".tools.websearch", "websearch"),
    "eval": (".eval", "eval"),
    "agents": (".agent", "agents"),
    "create_tool": (".agent", "create_tool"),
    "Skill": (".agent", "Skill"),
    "SkillRegistry": (".agent", "SkillRegistry"),
    "default_registry": (".agent", "default_registry"),
    "gen_image": (".tools.gen_img", "gen_image"),
    "edit_image": (".tools.gen_img", "edit_image"),
    "MemoryManager": (".helper.memory", "MemoryManager"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
