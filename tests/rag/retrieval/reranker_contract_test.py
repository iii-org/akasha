from dataclasses import dataclass
from importlib import import_module

import pytest

import akasha
from akasha.utils.optional_dependencies import OptionalDependencyError
from akasha.utils.search import rerank as rerank_module
from akasha.utils.search.rerank import rerank_documents

pytestmark = pytest.mark.unit


def _fake_model(prompt):
    return "answer"


def _fake_embeddings(texts):
    return [[0.0] for _ in texts]


def test_rag_exposes_reranker_independently_from_search_type():
    rag = akasha.RAG(
        model=_fake_model,
        embeddings=_fake_embeddings,
        search_type="auto",
        reranker="llm",
    )

    assert rag.search_type == "auto"
    assert rag.reranker == "llm"


def test_rag_uses_documented_reranker_defaults():
    rag = akasha.RAG(
        model=_fake_model,
        embeddings=_fake_embeddings,
    )

    assert rag.rerank_top_k == 5
    assert rag.reranker_model == "gemini:gemini-2.5-flash"


@dataclass
class _Document:
    page_content: str


def test_callable_reranker_receives_retrieved_documents():
    docs = [_Document("first"), _Document("second")]
    calls = []

    def reverse_reranker(query, candidates):
        calls.append((query, candidates))
        return list(reversed(candidates))

    result = rerank_documents("question", docs, reverse_reranker)

    assert [doc.page_content for doc in result] == ["second", "first"]
    assert calls == [("question", docs)]


def test_llm_reranker_uses_stable_candidate_ids(monkeypatch):
    docs = [_Document("first"), _Document("second")]
    prompts = []

    def fake_call_model(model_obj, prompt, verbose, keep_logs):
        prompts.append(prompt)
        return '{"order": [1, 0]}'

    monkeypatch.setattr(rerank_module, "_call_model", fake_call_model)

    result = rerank_documents("question", docs, "llm", model_obj=object())

    assert [doc.page_content for doc in result] == ["second", "first"]
    assert '"id": 0' in prompts[0]
    assert '"id": 1' in prompts[0]


def test_llm_reranker_rejects_non_integer_candidate_ids(monkeypatch):
    docs = [_Document("first"), _Document("second")]
    monkeypatch.setattr(
        rerank_module,
        "_call_model",
        lambda *args: '{"order": ["1", 0]}',
    )

    with pytest.raises(ValueError, match="every candidate id"):
        rerank_documents("question", docs, "llm", model_obj=object())


def test_local_bge_reranker_is_lazy_and_reorders_by_score(monkeypatch):
    docs = [_Document("first"), _Document("second")]
    calls = []

    def fake_score(model_name, pairs):
        calls.append((model_name, pairs))
        return [0.1, 0.9]

    monkeypatch.setattr(
        rerank_module,
        "validate_reranker_dependencies",
        lambda _reranker: None,
    )
    monkeypatch.setattr(rerank_module, "_score_with_local_bge", fake_score)

    result = rerank_documents("question", docs, "local:BAAI/bge-reranker-base")

    assert [doc.page_content for doc in result] == ["second", "first"]
    assert calls == [
        (
            "BAAI/bge-reranker-base",
            [["question", "first"], ["question", "second"]],
        )
    ]


def test_local_bge_reranker_reports_full_extra_when_dependencies_are_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        rerank_module,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("Local BGE reranking", "full")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        rerank_module.validate_reranker_dependencies("local:BAAI/bge-reranker-base")

    assert exc_info.value.extra == "full"


def test_rag_applies_reranker_after_retrieval(monkeypatch):
    rag_module = import_module("akasha.RAG.rag")
    retrieved = [_Document("first"), _Document("second")]
    calls = []

    def reverse_reranker(query, candidates):
        calls.append((query, candidates))
        return list(reversed(candidates))

    monkeypatch.setattr(rag_module, "get_retrivers", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag_module,
        "search_docs",
        lambda *args, **kwargs: (list(retrieved), 2, 2),
    )
    monkeypatch.setattr(rag_module.myTokenizer, "compute_tokens", lambda *args: 1)
    monkeypatch.setattr(rag_module, "get_doc_length", lambda *args: 1)

    source = rag_module.dbs()
    source.docs = ["placeholder"]
    source.metadatas = [{}]
    rag = rag_module.RAG(
        model=_fake_model,
        embeddings=_fake_embeddings,
        reranker=reverse_reranker,
    )

    assert rag(source, "question") == "answer"
    assert [doc.page_content for doc in rag.docs] == ["second", "first"]
    assert calls == [("question", retrieved)]


def test_rag_keeps_only_default_top_five_after_reranking(monkeypatch):
    rag_module = import_module("akasha.RAG.rag")
    retrieved = [_Document(f"document-{index}") for index in range(6)]

    monkeypatch.setattr(rag_module, "get_retrivers", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag_module,
        "search_docs",
        lambda *args, **kwargs: (list(retrieved), 6, 6),
    )
    monkeypatch.setattr(rag_module.myTokenizer, "compute_tokens", lambda *args: 1)
    monkeypatch.setattr(rag_module, "get_doc_length", lambda *args: 1)

    source = rag_module.dbs()
    source.docs = ["placeholder"]
    source.metadatas = [{}]
    rag = rag_module.RAG(
        model=_fake_model,
        embeddings=_fake_embeddings,
        reranker=lambda query, candidates: list(reversed(candidates)),
    )

    assert rag(source, "question") == "answer"
    assert [doc.page_content for doc in rag.docs] == [
        "document-5",
        "document-4",
        "document-3",
        "document-2",
        "document-1",
    ]
    assert rag.doc_length == 5
    assert rag.doc_tokens == 5


def test_llm_reranker_uses_default_dedicated_model_lazily(monkeypatch):
    rag_module = import_module("akasha.RAG.rag")
    atman_module = import_module("akasha.utils.atman")
    retrieved = [_Document("first"), _Document("second")]
    reranker_model_obj = object()
    created_models = []
    called_models = []

    monkeypatch.setattr(rag_module, "get_retrivers", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag_module,
        "search_docs",
        lambda *args, **kwargs: (list(retrieved), 2, 2),
    )
    monkeypatch.setattr(rag_module.myTokenizer, "compute_tokens", lambda *args: 1)
    monkeypatch.setattr(rag_module, "get_doc_length", lambda *args: 1)
    monkeypatch.setattr(
        rerank_module,
        "_call_model",
        lambda model_obj, *args, **kwargs: (
            called_models.append(model_obj) or '{"order": [1, 0]}'
        ),
    )

    source = rag_module.dbs()
    source.docs = ["placeholder"]
    source.metadatas = [{}]
    rag = rag_module.RAG(
        model=_fake_model,
        embeddings=_fake_embeddings,
        reranker="llm",
    )

    monkeypatch.setattr(
        atman_module,
        "handle_model",
        lambda model, *args, **kwargs: (
            created_models.append(model) or reranker_model_obj
        ),
    )
    assert created_models == []
    assert rag(source, "question") == "answer"
    assert created_models == ["gemini:gemini-2.5-flash"]
    assert called_models == [reranker_model_obj]


def test_rag_call_can_override_reranker_model_and_top_k(monkeypatch):
    rag_module = import_module("akasha.RAG.rag")
    atman_module = import_module("akasha.utils.atman")
    retrieved = [_Document(f"document-{index}") for index in range(4)]
    reranker_model_obj = object()
    created_models = []

    monkeypatch.setattr(rag_module, "get_retrivers", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag_module,
        "search_docs",
        lambda *args, **kwargs: (list(retrieved), 4, 4),
    )
    monkeypatch.setattr(rag_module.myTokenizer, "compute_tokens", lambda *args: 1)
    monkeypatch.setattr(rag_module, "get_doc_length", lambda *args: 1)
    monkeypatch.setattr(
        rerank_module,
        "_call_model",
        lambda *args, **kwargs: '{"order": [3, 2, 1, 0]}',
    )

    source = rag_module.dbs()
    source.docs = ["placeholder"]
    source.metadatas = [{}]
    rag = rag_module.RAG(model=_fake_model, embeddings=_fake_embeddings)
    monkeypatch.setattr(
        atman_module,
        "handle_model",
        lambda model, *args, **kwargs: (
            created_models.append(model) or reranker_model_obj
        ),
    )

    assert (
        rag(
            source,
            "question",
            reranker="llm",
            reranker_model="gemini:custom-reranker",
            rerank_top_k=2,
        )
        == "answer"
    )
    assert created_models == ["gemini:custom-reranker"]
    assert [doc.page_content for doc in rag.docs] == ["document-3", "document-2"]
