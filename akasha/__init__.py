from akasha.utils.logging_config import configure_logging
import os

# Optional automatic logging configuration.
# Enable by setting AKASHA_AUTO_CONFIGURE_LOGGING to "1", "true", or "yes".
os.environ["CHROMA_TELEMETRY_OPT_OUT"] = "True"
if os.getenv("AKASHA_AUTO_CONFIGURE_LOGGING", "").lower() in {"1", "true", "yes"}:
    configure_logging(verbose=True, keep_logs=False)

from .RAG.rag import RAG
from .tools.ask import ask
from .helper.memory import MemoryManager
from .tools.summary import summary
from .tools.websearch import websearch
from .tools.gen_img import gen_image, edit_image
from .eval import eval
from .agent import agents
from .agent import create_tool

__all__ = [
    "RAG",
    "ask",
    "summary",
    "websearch",
    "eval",
    "agents",
    "create_tool",
    "gen_image",
    "edit_image",
    "MemoryManager",
]
