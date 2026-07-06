import pytest
from langchain_core.documents import Document

from akasha.utils.search import search_doc

pytestmark = pytest.mark.unit


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def _get_relevant_documents(self, query):
        return list(self.docs)


def test_merge_docs_deduplicates_and_stops_on_token_limit(monkeypatch):
    monkeypatch.setattr(search_doc, "get_doc_length", lambda language, text: len(text.split()))
    monkeypatch.setattr(search_doc.myTokenizer, "compute_tokens", lambda text, model: len(text.split()))
    docs_a = [Document(page_content="one two"), Document(page_content="shared doc")]
    docs_b = [Document(page_content="shared doc"), Document(page_content="three four five")]

    docs, doc_len, tokens = search_doc._merge_docs(
        [docs_a, docs_b],
        topK=3,
        language="en",
        max_input_tokens=4,
        model="openai:gpt-4o",
    )

    assert [doc.page_content for doc in docs] == ["one two", "shared doc"]
    assert doc_len == 4
    assert tokens == 4


def test_search_docs_uses_auto_helpers(monkeypatch):
    monkeypatch.setattr(
        search_doc,
        "get_relevant_doc_auto",
        lambda retrievers, query: [Document(page_content="alpha"), Document(page_content="beta")],
    )
    monkeypatch.setattr(search_doc, "get_doc_length", lambda language, text: 1)
    monkeypatch.setattr(search_doc.myTokenizer, "compute_tokens", lambda text, model: 1)

    docs, docs_len, tokens = search_doc.search_docs(
        [_FakeRetriever([])],
        "question",
        search_type="auto",
        language="en",
        max_input_tokens=10,
    )

    assert [doc.page_content for doc in docs] == ["alpha", "beta"]
    assert docs_len == 2
    assert tokens == 2


def test_search_docs_merges_multiple_retrievers(monkeypatch):
    monkeypatch.setattr(search_doc, "get_doc_length", lambda language, text: 1)
    monkeypatch.setattr(search_doc.myTokenizer, "compute_tokens", lambda text, model: 1)
    retrievers = [
        _FakeRetriever([Document(page_content="alpha"), Document(page_content="shared")]),
        _FakeRetriever([Document(page_content="shared"), Document(page_content="beta")]),
    ]

    docs, docs_len, tokens = search_doc.search_docs(
        retrievers,
        "question",
        search_type="merge",
        language="en",
        max_input_tokens=10,
    )

    assert [doc.page_content for doc in docs] == ["alpha", "shared", "beta"]
    assert docs_len == 3
    assert tokens == 3


def test_retri_docs_uses_auto_and_deduplicates(monkeypatch):
    monkeypatch.setattr(
        search_doc,
        "get_relevant_doc_auto_rerank",
        lambda retrievers, query, topK: [
            Document(page_content="alpha"),
            Document(page_content="alpha"),
            Document(page_content="beta"),
        ],
    )

    docs = search_doc.retri_docs([_FakeRetriever([])], "question", "auto_rerank", topK=2)

    assert [doc.page_content for doc in docs] == ["alpha", "beta"]
