from importlib import import_module
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .RAG.rag import RAG
    from .agent import Skill, SkillRegistry, agents, create_tool, default_registry, normalize_mcp_result, normalize_mcp_tools
    from .embeddings import (
        create_embeddings,
        describe_embeddings,
        embed_documents,
        embed_query,
    )
    from .eval import eval
    from .helper.memory import MemoryManager
    from .tools.ask import ask
    from .tools.gen_img import edit_image, gen_image
    from .tools.summary import summary
    from .tools.websearch import websearch

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
    "normalize_mcp_result",
    "normalize_mcp_tools",
    "Skill",
    "SkillRegistry",
    "default_registry",
    "gen_image",
    "edit_image",
    "MemoryManager",
    "create_embeddings",
    "describe_embeddings",
    "embed_documents",
    "embed_query",
]

_LAZY_IMPORTS = {
    "RAG": (".RAG.rag", "RAG"),
    "ask": (".tools.ask", "ask"),
    "summary": (".tools.summary", "summary"),
    "websearch": (".tools.websearch", "websearch"),
    "eval": (".eval", "eval"),
    "agents": (".agent", "agents"),
    "create_tool": (".agent", "create_tool"),
    "normalize_mcp_result": (".agent", "normalize_mcp_result"),
    "normalize_mcp_tools": (".agent", "normalize_mcp_tools"),
    "Skill": (".agent", "Skill"),
    "SkillRegistry": (".agent", "SkillRegistry"),
    "default_registry": (".agent", "default_registry"),
    "gen_image": (".tools.gen_img", "gen_image"),
    "edit_image": (".tools.gen_img", "edit_image"),
    "MemoryManager": (".helper.memory", "MemoryManager"),
    "create_embeddings": (".embeddings", "create_embeddings"),
    "describe_embeddings": (".embeddings", "describe_embeddings"),
    "embed_documents": (".embeddings", "embed_documents"),
    "embed_query": (".embeddings", "embed_query"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public APIs to defer optional provider imports."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
