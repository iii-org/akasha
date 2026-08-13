"""Stage-by-stage live RAG checks for the Chroma-to-answer pipeline."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values, load_dotenv

import akasha
from akasha.helper.preprocess_prompts import merge_history_and_prompt
from akasha.helper.run_llm import call_model
from akasha.utils.db.db_structure import get_storage_directory
from akasha.utils.db.load_db import load_db_by_chroma_name
from akasha.utils.search.retrievers.base import get_retrivers
from akasha.utils.search.search_doc import search_docs


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / "tests" / ".env"
RAG_FILE = REPO_ROOT / "tests" / "tests_data" / "rag_smoke" / "single_fact.txt"
RUN_LIVE = os.getenv("RUN_RAG_PIPELINE", "").lower() in {"1", "true", "yes"}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_RAG_PIPELINE=1 to enable staged live RAG checks",
    ),
]


def _require_openai() -> None:
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if not (os.getenv("OPENAI_API_KEY") or values.get("OPENAI_API_KEY")):
        pytest.skip("OPENAI_API_KEY is not configured")


@pytest.fixture(scope="module")
def rag():
    _require_openai()
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    return akasha.RAG(
        model="openai:gpt-5.4",
        embeddings="openai:text-embedding-3-small",
        chunk_size=1000,
        search_type="auto",
        max_output_tokens=128,
        keep_logs=True,
        env_file="",
    )


@pytest.fixture(scope="module")
def built_db(rag):
    print("[stage 1] build/load Chroma from source", flush=True)
    rag._get_db(RAG_FILE)
    assert rag.db.get_docs(), "Chroma build returned no documents"
    return rag.db


def test_rag_stage_1_reload_chroma(rag, built_db):
    """A built Chroma store can be opened again and read independently."""
    storage_dir = get_storage_directory(
        RAG_FILE.parent,
        rag.chunk_size,
        "openai",
        "text-embedding-3-small",
    )
    print(f"[stage 1] reload: {storage_dir}", flush=True)
    reloaded, ignored = load_db_by_chroma_name(storage_dir)

    assert not ignored
    assert reloaded.get_docs()
    assert any("RAG-7319-TAIPEI" in text for text in reloaded.get_docs())


def test_rag_stage_2_search_relevant_document(rag, built_db):
    """The retriever returns the document containing the verification code."""
    print("[stage 2] create retrievers and search", flush=True)
    retrievers = get_retrivers(
        built_db,
        rag.embeddings_obj,
        rag.threshold,
        rag.search_type,
        rag.env_file,
    )
    docs, doc_length, doc_tokens = search_docs(
        retrievers,
        "What is the verification code?",
        rag.model,
        rag.max_input_tokens,
        rag.search_type,
        rag.language,
    )

    assert retrievers
    assert docs
    assert doc_length > 0
    assert doc_tokens > 0
    assert any("RAG-7319-TAIPEI" in doc.page_content for doc in docs)


def test_rag_stage_3_chat_model_answers_from_retrieved_document(rag, built_db):
    """The chat model receives retrieved context and returns text."""
    print("[stage 3] retrieve context and call chat model", flush=True)
    retrievers = get_retrivers(
        built_db,
        rag.embeddings_obj,
        rag.threshold,
        rag.search_type,
        rag.env_file,
    )
    docs, _, _ = search_docs(
        retrievers,
        "What is the verification code?",
        rag.model,
        rag.max_input_tokens,
        rag.search_type,
        rag.language,
    )
    rag.docs = docs
    prompt = merge_history_and_prompt(
        [],
        rag.system_prompt,
        rag._format_docs() + "User question: What is the verification code?",
        rag.prompt_format_type,
        model=rag.model,
    )
    response = call_model(rag.model_obj, prompt, verbose=False, keep_logs=True)

    assert isinstance(response, str)
    assert response.strip()
    assert "RAG-7319-TAIPEI" in response


def test_rag_stage_4_cleanup_is_executable(rag, built_db):
    """Release RAG-owned references and run collection before pytest exits."""
    print("[stage 4] release RAG references and collect", flush=True)
    rag.db = None
    rag.docs = []
    rag.embeddings_obj = None
    rag.model_obj = None
    gc.collect()
