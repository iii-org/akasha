import pytest


@pytest.mark.helper
def test_get_retrievers_bm25(monkeypatch: pytest.MonkeyPatch):
    from akasha.utils.db.db_structure import dbs
    from akasha.utils.search.retrievers import base as retriever_base

    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda *args, **kwargs: (object(), "dummy"),
    )

    db = dbs([])
    db.docs = ["foo bar", "baz qux", "foo baz"]
    db.metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

    retrievers = retriever_base.get_retrivers(
        db=db,
        embeddings="gemini:gemini-embedding-001",
        search_type="bm25",
    )

    assert len(retrievers) == 1
    docs, _scores = retrievers[0]._gs("foo")
    assert docs


@pytest.mark.helper
def test_get_retrievers_tfidf(monkeypatch: pytest.MonkeyPatch):
    from akasha.utils.db.db_structure import dbs
    from akasha.utils.search.retrievers import base as retriever_base

    monkeypatch.setattr(
        retriever_base,
        "handle_embeddings_and_name",
        lambda *args, **kwargs: (object(), "dummy"),
    )

    db = dbs([])
    db.docs = ["alpha beta", "beta gamma", "gamma delta"]
    db.metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

    retrievers = retriever_base.get_retrivers(
        db=db,
        embeddings="gemini:gemini-embedding-001",
        search_type="tfidf",
    )

    assert len(retrievers) == 1
    docs, _scores = retrievers[0]._gs("beta")
    assert docs
