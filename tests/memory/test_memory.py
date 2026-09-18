from types import SimpleNamespace

import pytest

from akasha.helper import memory as memory_module


pytestmark = pytest.mark.unit


def test_memory_search_uses_core_auto_retrieval(monkeypatch):
    manager = memory_module.MemoryManager.__new__(memory_module.MemoryManager)
    manager.db = SimpleNamespace()
    manager.embeddings_obj = SimpleNamespace()
    manager.verbose = False
    calls = {}

    def fake_get_retrivers(db, embeddings, threshold, search_type, env_file):
        calls["get_retrivers"] = search_type
        return ["retriever"]

    def fake_retri_docs(retrievers, query, search_type, top_k):
        calls["retri_docs"] = search_type
        return [SimpleNamespace(page_content="remembered")]

    monkeypatch.setattr(memory_module, "get_retrivers", fake_get_retrivers)
    monkeypatch.setattr(memory_module, "retri_docs", fake_retri_docs)

    assert manager.search_memory("query", top_k=3) == ["remembered"]
    assert calls == {
        "get_retrivers": "auto",
        "retri_docs": "auto",
    }
