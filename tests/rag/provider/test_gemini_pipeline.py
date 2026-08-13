"""Gemini embedding -> Chroma -> retrieval -> Gemini answer smoke test."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values, load_dotenv

import akasha
from akasha.helper.preprocess_prompts import merge_history_and_prompt
from akasha.helper.run_llm import call_model
from akasha.utils.db.db_structure import get_storage_directory
from akasha.utils.db.load_db import load_db_by_chroma_name
from akasha.utils.search.retrievers.base import get_retrivers
from akasha.utils.search.search_doc import search_docs
from akasha.helper.base import separate_name
from tests.support.paths import REPO_ROOT, TEST_ENV_FILE, RAG_DATA_ROOT


ENV_FILE = TEST_ENV_FILE
MODEL_MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"
RAG_FILE = RAG_DATA_ROOT / "single_fact.txt"
RUN_LIVE = os.getenv("RUN_GEMINI_RAG", "").lower() in {"1", "true", "yes"}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_GEMINI_RAG=1 to enable the live Gemini RAG pipeline",
    ),
]


def test_gemini_embedding_chroma_search_and_answer():
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if not (os.getenv("GEMINI_API_KEY") or values.get("GEMINI_API_KEY")):
        pytest.skip("GEMINI_API_KEY is not configured")
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)

    manifest = yaml.safe_load(MODEL_MANIFEST.read_text(encoding="utf-8"))
    embedding_name = next(
        item["id"] for item in manifest["embeddings"] if item["provider"] == "gemini"
    )
    embedding_type, embedding_model = separate_name(embedding_name)
    assert embedding_name == "gemini:gemini-embedding-2"

    print("[gemini setup] initialize RAG", flush=True)
    rag = akasha.RAG(
        model="gemini:gemini-2.5-flash",
        embeddings=embedding_name,
        chunk_size=1000,
        search_type="auto",
        max_output_tokens=128,
        keep_logs=True,
        env_file="",
    )

    print("[gemini stage 1] real embedding output -> Chroma", flush=True)
    rag._get_db(RAG_FILE)
    assert rag.db.get_docs()
    assert any("RAG-7319-TAIPEI" in text for text in rag.db.get_docs())

    storage_dir = get_storage_directory(
        RAG_FILE.parent,
        rag.chunk_size,
        embedding_type,
        embedding_model,
    )
    print(f"[gemini stage 2] reload Chroma: {storage_dir}", flush=True)
    reloaded_db, ignored = load_db_by_chroma_name(storage_dir)
    assert not ignored
    assert reloaded_db.get_docs()

    print("[gemini stage 3] search reloaded Chroma", flush=True)
    print("[gemini stage 3a] create retrievers", flush=True)
    retrievers = get_retrivers(
        reloaded_db,
        rag.embeddings_obj,
        rag.threshold,
        rag.search_type,
        rag.env_file,
    )
    print("[gemini stage 3b] run retriever search", flush=True)
    docs, doc_length, doc_tokens = search_docs(
        retrievers,
        "What is the verification code?",
        rag.model,
        rag.max_input_tokens,
        rag.search_type,
        rag.language,
    )
    print("[gemini stage 3c] search completed", flush=True)
    assert docs
    assert doc_length > 0
    assert doc_tokens > 0
    assert any("RAG-7319-TAIPEI" in doc.page_content for doc in docs)

    print("[gemini stage 4] retrieved context -> real Gemini chat model", flush=True)
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

    print("[gemini stage 5] cleanup", flush=True)
    rag.db = None
    rag.docs = []
    rag.embeddings_obj = None
    rag.model_obj = None
    reloaded_db = None
    retrievers = None
    gc.collect()
