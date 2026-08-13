"""Opt-in RAG smoke tests covering the public file/path contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values, load_dotenv

import akasha
from tests.support.paths import REPO_ROOT, TEST_ENV_FILE, RAG_DATA_ROOT


ENV_FILE = TEST_ENV_FILE
RUN_LIVE = os.getenv("RUN_RAG_SMOKE", "").strip().lower() in {"1", "true", "yes"}
RAG_DATA = RAG_DATA_ROOT


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_RAG_SMOKE=1 to enable real RAG smoke tests",
    ),
]


def _configured(key: str) -> bool:
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    return bool(os.getenv(key) or values.get(key))


def _load_env() -> str:
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    return ""


def _assert_json_safe(value) -> None:
    json.dumps(value, ensure_ascii=False)


def _rag(model: str, embeddings: str, env_file: str):
    return akasha.RAG(
        model=model,
        embeddings=embeddings,
        keep_logs=True,
        max_output_tokens=128,
        env_file=env_file,
    )


def test_openai_rag_file_contract():
    """OpenAI RAG must ingest one file and return a serializable contract."""
    if not _configured("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    rag = _rag("openai:gpt-5.4", "openai:text-embedding-3-small", _load_env())
    response = rag(RAG_DATA / "single_fact.txt", "What is the verification code?")

    assert isinstance(response, str) and response.strip()
    assert rag.docs
    assert any("RAG-7319-TAIPEI" in doc.page_content for doc in rag.docs)
    assert rag.logs
    _assert_json_safe(rag.logs)


def test_gemini_rag_directory_contract():
    """Gemini RAG must ingest a directory and retrieve the relevant file."""
    if not _configured("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")

    rag = _rag("gemini:gemini-3.5-flash", "gemini:gemini-embedding-2", _load_env())
    response = rag(RAG_DATA / "directory", "Which protocol does Device Beta use?")

    assert isinstance(response, str) and response.strip()
    assert rag.docs
    assert any("ORBIT-882" in doc.page_content for doc in rag.docs)
    assert rag.logs
    _assert_json_safe(rag.logs)


@pytest.mark.skipif(sys.platform != "win32", reason="requires a Windows runner")
def test_rag_windows_absolute_path_contract():
    """A Windows runner must accept an absolute Path object as file input."""
    if not _configured("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    rag = _rag("openai:gpt-5.4", "openai:text-embedding-3-small", _load_env())
    response = rag(RAG_DATA / "single_fact.txt", "Return the verification code.")

    assert isinstance(response, str) and response.strip()
    assert rag.docs
