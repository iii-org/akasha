"""Compatibility helpers for optional Chroma dependencies."""

from __future__ import annotations


def get_chroma_components():
    try:
        from chromadb.config import Settings
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise ImportError(
            "Feature requiring 'chromadb/langchain-chroma' is not installed. "
            "Please install with: pip install akasha-terminal[light]"
        ) from exc

    return Chroma, Settings
