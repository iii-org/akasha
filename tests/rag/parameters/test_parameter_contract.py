"""Deterministic contract coverage for the public RAG parameters.

These tests stop at the model, embedding, database, and retriever seams.  The
provider-backed smoke tests remain responsible for validating real services;
this module verifies that RAG itself forwards user configuration correctly.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.documents import Document

from akasha.RAG.rag import RAG


pytestmark = pytest.mark.unit

atman_module = importlib.import_module("akasha.utils.atman")
rag_module = importlib.import_module("akasha.RAG.rag")


def _build_rag(monkeypatch, **kwargs) -> RAG:
    model_calls = []
    embedding_calls = []

    def fake_handle_model(*args):
        model_calls.append(args)
        return object()

    def fake_handle_embeddings(*args):
        embedding_calls.append(args)
        return object()

    monkeypatch.setattr(atman_module, "handle_model", fake_handle_model)
    monkeypatch.setattr(atman_module, "handle_embeddings", fake_handle_embeddings)
    monkeypatch.setattr(atman_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(atman_module, "require_mlflow", lambda: None)

    rag = RAG(
        model="fake:model",
        embeddings="fake:embedding",
        **kwargs,
    )
    rag._test_model_calls = model_calls
    rag._test_embedding_calls = embedding_calls
    return rag


def _patch_local_pipeline(monkeypatch, rag: RAG, *, stream: bool = False):
    document = Document(
        page_content="The verification code is RAG-CONTRACT-7319.",
        metadata={"source": "fixture.txt", "page": 1},
    )
    captured = {}

    def fake_get_db(source):
        captured["data_source"] = source
        rag.db = object()

    def fake_get_retrievers(*args):
        captured["retriever_args"] = args
        return [object()]

    def fake_search_docs(*args):
        captured["search_args"] = args
        return [document], 8, 8

    def fake_merge(history, system_prompt, prompt, prompt_format_type, **kwargs):
        captured["prompt_args"] = {
            "history": history,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "prompt_format_type": prompt_format_type,
            "model": kwargs.get("model"),
        }
        return "formatted-rag-prompt"

    def fake_call_model(model_obj, prompt, verbose, **kwargs):
        captured["model_call"] = {
            "model_obj": model_obj,
            "prompt": prompt,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return "RAG-CONTRACT-7319"

    monkeypatch.setattr(rag, "_get_db", fake_get_db)
    monkeypatch.setattr(rag, "_check_db", lambda: True)
    monkeypatch.setattr(rag_module, "get_retrivers", fake_get_retrievers)
    monkeypatch.setattr(rag_module, "search_docs", fake_search_docs)
    monkeypatch.setattr(rag_module, "merge_history_and_prompt", fake_merge)
    monkeypatch.setattr(rag_module, "call_model", fake_call_model)
    monkeypatch.setattr(
        rag_module.myTokenizer,
        "compute_tokens",
        lambda _text, _model: 1,
    )

    if stream:
        def fake_stream_model(model_obj, prompt, verbose, **kwargs):
            captured["stream_call"] = {
                "model_obj": model_obj,
                "prompt": prompt,
                "verbose": verbose,
                "kwargs": kwargs,
            }
            return iter(("RAG-", "CONTRACT-7319"))

        monkeypatch.setattr(rag_module, "call_stream_model", fake_stream_model)
    return captured


def test_rag_constructor_forwards_and_preserves_public_parameters(monkeypatch):
    rag = _build_rag(
        monkeypatch,
        chunk_size=321,
        search_type="merge",
        max_input_tokens=777,
        max_output_tokens=123,
        temperature=0.25,
        threshold=0.4,
        language="en",
        record_exp="contract-test",
        system_prompt="Answer only from the supplied documents.",
        prompt_format_type="chat_gpt",
        keep_logs=True,
        use_chroma=True,
        stream=True,
        verbose=True,
        env_file="tests/.env.contract",
    )

    assert rag.chunk_size == 321
    assert rag.search_type == "merge"
    assert rag.max_input_tokens == 777
    assert rag.max_output_tokens == 123
    assert rag.temperature == 0.25
    assert rag.threshold == 0.4
    assert rag.language == "en"
    assert rag.record_exp == "contract-test"
    assert rag.system_prompt == "Answer only from the supplied documents."
    assert rag.prompt_format_type == "chat_gpt"
    assert rag.keep_logs is True
    assert rag.use_chroma is True
    assert rag.stream is True
    assert rag.verbose is True
    assert rag.env_file == "tests/.env.contract"

    assert rag._test_model_calls[0][0] == "fake:model"
    assert rag._test_model_calls[0][2:6] == (
        0.25,
        123,
        "tests/.env.contract",
        False,
    )
    assert rag._test_embedding_calls == [
        ("fake:embedding", True, "tests/.env.contract")
    ]


def test_rag_call_forwards_sources_retrieval_prompt_history_and_generation(monkeypatch):
    rag = _build_rag(
        monkeypatch,
        chunk_size=321,
        search_type="auto",
        max_input_tokens=777,
        max_output_tokens=123,
        temperature=0.25,
        threshold=0.4,
        language="en",
        system_prompt="Use the source only.",
        prompt_format_type="chat_gpt",
        verbose=True,
        env_file="fixture.env",
    )
    captured = _patch_local_pipeline(monkeypatch, rag)

    source = Path("tests/data/rag/single_fact.txt")
    history = ["user: previous question", "assistant: previous answer"]
    response = rag(source, "What is the verification code?", history_messages=history)

    assert response == "RAG-CONTRACT-7319"
    assert captured["data_source"] == source
    assert captured["retriever_args"][1] is rag.embeddings_obj
    assert captured["retriever_args"][2:5] == (
        rag.threshold,
        rag.search_type,
        rag.env_file,
    )
    assert captured["search_args"][1] == "What is the verification code?"
    assert captured["search_args"][3] == rag.max_input_tokens - rag.prompt_tokens
    assert captured["search_args"][5] == rag.language
    assert captured["prompt_args"]["history"] == history
    assert captured["prompt_args"]["system_prompt"] == "Use the source only."
    assert captured["prompt_args"]["prompt_format_type"] == "chat_gpt"
    assert captured["prompt_args"]["model"] == rag.model
    assert captured["model_call"]["prompt"] == "formatted-rag-prompt"
    assert captured["model_call"]["verbose"] is True
    assert captured["model_call"]["kwargs"] == {"keep_logs": False}


def test_rag_stream_returns_chunks_and_accumulates_response(monkeypatch):
    rag = _build_rag(
        monkeypatch,
        stream=True,
        max_input_tokens=100,
    )
    captured = _patch_local_pipeline(monkeypatch, rag, stream=True)

    chunks = list(rag(Path("tests/data/rag/single_fact.txt"), "Give the code."))

    assert chunks == ["RAG-", "CONTRACT-7319"]
    assert rag.response == "RAG-CONTRACT-7319"
    assert captured["stream_call"]["prompt"] == "formatted-rag-prompt"
    assert captured["stream_call"]["kwargs"] == {"keep_logs": False}


def test_rag_use_chroma_loads_existing_store(monkeypatch, tmp_path):
    rag = _build_rag(monkeypatch, use_chroma=True)
    loaded_db = object()
    calls = []

    def fake_load(name):
        calls.append(name)
        return loaded_db, ["ignored.txt"]

    monkeypatch.setattr(atman_module, "load_db_by_chroma_name", fake_load)

    source = tmp_path / "existing-chroma"
    rag._get_db(source)

    assert calls == [source]
    assert rag.db is loaded_db
    assert rag.ignored_files == ["ignored.txt"]


def test_rag_rejects_prompt_over_max_input_tokens_before_retrieval(monkeypatch):
    rag = _build_rag(monkeypatch, max_input_tokens=1)
    captured = _patch_local_pipeline(monkeypatch, rag)
    monkeypatch.setattr(rag_module.myTokenizer, "compute_tokens", lambda *_: 100)

    with pytest.raises(ValueError, match="max_input_tokens"):
        rag(Path("tests/data/rag/single_fact.txt"), "A prompt that is too long")

    assert "retriever_args" not in captured
