"""Live RAG contract shared by the supported remote providers."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

import akasha
from tests.support.live import load_test_env, require_keys
from tests.support.paths import RAG_DATA_ROOT


@dataclass(frozen=True)
class RAGProviderCase:
    model: str
    embeddings: str
    required_keys: tuple[str, ...]


PROVIDER_CASES = [
    pytest.param(
        RAGProviderCase(
            model="openai:gpt-5.4",
            embeddings="openai:text-embedding-3-small",
            required_keys=("OPENAI_API_KEY",),
        ),
        id="openai",
    ),
    pytest.param(
        RAGProviderCase(
            model="gemini:gemini-2.5-flash",
            embeddings="gemini:gemini-embedding-2",
            required_keys=("GEMINI_API_KEY",),
        ),
        id="gemini",
    ),
    pytest.param(
        RAGProviderCase(
            model="anthropic:claude-sonnet-4-6",
            embeddings="openai:text-embedding-3-small",
            required_keys=("ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
        ),
        id="anthropic-with-openai-embeddings",
    ),
]


pytestmark = [
    pytest.mark.live,
    pytest.mark.contract,
    pytest.mark.provider,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


@pytest.mark.parametrize("case", PROVIDER_CASES)
def test_rag_provider_returns_grounded_serializable_result(case: RAGProviderCase):
    """Each provider must ingest, retrieve, answer, and expose stable logs."""

    require_keys(*case.required_keys)
    rag = akasha.RAG(
        model=case.model,
        embeddings=case.embeddings,
        keep_logs=True,
        max_output_tokens=128,
        env_file=load_test_env(),
    )

    response = rag(
        RAG_DATA_ROOT / "single_fact.txt",
        "Return the verification code from the document.",
    )

    assert isinstance(response, str) and response.strip()
    assert "RAG-7319-TAIPEI" in response
    assert rag.docs
    assert any("RAG-7319-TAIPEI" in doc.page_content for doc in rag.docs)
    assert rag.logs
    json.dumps(rag.logs, ensure_ascii=False)
