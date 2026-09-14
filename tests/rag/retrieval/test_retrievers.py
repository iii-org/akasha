import pytest

from akasha.utils.db.db_structure import dbs
from akasha.utils.optional_dependencies import OptionalDependencyError
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
    monkeypatch.setattr(
        retriever_base.myTFIDFRetriever, "from_documents", record("tfidf")
    )
    monkeypatch.setattr(retriever_base.myKNNRetriever, "from_db", record("knn"))
    monkeypatch.setattr(
        retriever_base.myBM25Retriever, "from_documents", record("bm25")
    )
    fake_faiss_retriever = type(
        "FakeFaissRetriever",
        (),
        {"from_db": staticmethod(record("faiss"))},
    )
    monkeypatch.setattr(
        retriever_base,
        "_get_faiss_retriever_class",
        lambda: fake_faiss_retriever,
    )

    fake_db = _FakeDB()

    merge_result = retriever_base.get_retrivers(fake_db, "embed", search_type="merge")
    auto_result = retriever_base.get_retrivers(fake_db, "embed", search_type="auto")
    faiss_result = retriever_base.get_retrivers(fake_db, "embed", search_type="faiss")

    assert merge_result == ["mmr-retriever", "svm-retriever", "tfidf-retriever"]
    assert auto_result == ["knn-retriever", "bm25-retriever"]
    assert faiss_result == ["faiss-retriever"]
    assert {name for name, _, _ in calls} >= {
        "mmr",
        "svm",
        "tfidf",
        "knn",
        "bm25",
        "faiss",
    }


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


def test_get_retrievers_fails_cleanly_when_rerank_support_is_missing(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )
    monkeypatch.setattr(
        retriever_base,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("Local rerank retrieval", "full")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        retriever_base.get_retrivers(_FakeDB(), "embed", search_type="rerank")

    assert exc_info.value.extra == "full"


def test_get_retrievers_raises_on_unknown_search_type(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda embeddings, _verbose, _env_file: ("embed-obj", "fake:embed"),
    )

    with pytest.raises(ValueError, match="cannot find search type mystery"):
        retriever_base.get_retrivers(_FakeDB(), "embed", search_type="mystery")


def test_missing_faiss_points_to_optional_profile(monkeypatch):
    monkeypatch.setattr(
        retriever_base,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("FAISS retrieval", "faiss")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        retriever_base._get_faiss_retriever_class()

    assert exc_info.value.extra == "faiss"
