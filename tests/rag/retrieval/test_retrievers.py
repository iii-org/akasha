import builtins

import pytest

from akasha.utils.db.db_structure import dbs
from akasha.utils.search.retrievers import base as retriever_base

pytestmark = pytest.mark.unit


class _FakeDB(dbs):
    def __init__(self):
        super().__init__()
        self.docs = ["doc-a", "doc-b"]
        self.metadatas = [{}, {}]


def test_get_retrievers_builds_expected_retriever_types(monkeypatch):
    calls = []

    def record(name):
        def _factory(*args, **kwargs):
            calls.append((name, args, kwargs))
            return f"{name}-retriever"

        return _factory

    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )
    monkeypatch.setattr(retriever_base.myMMRRetriever, "from_db", record("mmr"))
    monkeypatch.setattr(retriever_base.mySVMRetriever, "from_db", record("svm"))
    monkeypatch.setattr(retriever_base.myTFIDFRetriever, "from_documents", record("tfidf"))
    monkeypatch.setattr(retriever_base.myKNNRetriever, "from_db", record("knn"))
    monkeypatch.setattr(retriever_base.myBM25Retriever, "from_documents", record("bm25"))
    monkeypatch.setattr(retriever_base.myFAISSRetriever, "from_db", record("faiss"))

    fake_db = _FakeDB()

    merge_result = retriever_base.get_retrivers(fake_db, "embed", search_type="merge")
    auto_result = retriever_base.get_retrivers(fake_db, "embed", search_type="auto")
    faiss_result = retriever_base.get_retrivers(fake_db, "embed", search_type="faiss")

    assert merge_result == ["mmr-retriever", "svm-retriever", "tfidf-retriever"]
    assert auto_result == ["knn-retriever", "bm25-retriever"]
    assert faiss_result == ["faiss-retriever"]
    assert {name for name, _, _ in calls} >= {"mmr", "svm", "tfidf", "knn", "bm25", "faiss"}


def test_get_retrievers_supports_custom_callable(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )
    monkeypatch.setattr(
        retriever_base.customRetriever,
        "from_db",
        lambda db_obj, embeddings, search_type, topK, threshold: (
            db_obj,
            embeddings,
            search_type,
            topK,
            threshold,
        ),
    )

    def custom_search():
        return []

    result = retriever_base.get_retrivers(_FakeDB(), "embed", search_type=custom_search)

    assert result[0][1] == "embed-obj"
    assert result[0][2] is custom_search


def test_get_retrievers_warns_when_rerank_support_is_missing(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError):
        retriever_base.get_retrivers(_FakeDB(), "embed", search_type="rerank")


def test_get_retrievers_raises_on_unknown_search_type(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )

    with pytest.raises(ValueError, match="cannot find search type mystery"):
        retriever_base.get_retrivers(_FakeDB(), "embed", search_type="mystery")
