"""Convenience functions for creating and using embedding adapters."""

from typing import Any, Callable, Sequence

from langchain_core.embeddings import Embeddings

from akasha.helper.base import get_embedding_type_and_name
from akasha.helper.handle_objects import handle_embeddings
from akasha.utils.base import DEFAULT_EMBED

EmbeddingSpec = str | Embeddings | Callable[..., Any]


def create_embeddings(
    embeddings: EmbeddingSpec = DEFAULT_EMBED,
    *,
    verbose: bool = False,
    env_file: str = "",
) -> Embeddings:
    """Create an embedding adapter from a model name or implementation.

    Args:
        embeddings: Alias such as ``"openai:text-embedding-3-small"``, an
            existing LangChain ``Embeddings`` object, or a custom callable.
        verbose: Whether to print provider selection information.
        env_file: Optional dotenv file containing provider credentials.

    Returns:
        A LangChain ``Embeddings`` implementation.
    """
    return handle_embeddings(embeddings, verbose=verbose, env_file=env_file)


def describe_embeddings(embeddings: EmbeddingSpec) -> tuple[str, str]:
    """Return the provider type and model name for an embedding adapter."""
    return get_embedding_type_and_name(embeddings)


def embed_documents(
    texts: Sequence[str],
    embeddings: EmbeddingSpec = DEFAULT_EMBED,
    *,
    verbose: bool = False,
    env_file: str = "",
) -> list[list[float]]:
    """Embed multiple documents and return one vector per document."""
    adapter = create_embeddings(embeddings, verbose=verbose, env_file=env_file)
    return adapter.embed_documents(list(texts))


def embed_query(
    text: str,
    embeddings: EmbeddingSpec = DEFAULT_EMBED,
    *,
    verbose: bool = False,
    env_file: str = "",
) -> list[float]:
    """Embed one query and return its vector representation."""
    adapter = create_embeddings(embeddings, verbose=verbose, env_file=env_file)
    return adapter.embed_query(text)


__all__ = [
    "EmbeddingSpec",
    "create_embeddings",
    "describe_embeddings",
    "embed_documents",
    "embed_query",
]
