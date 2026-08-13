"""Unit coverage for the RAG document-input contract.

These tests deliberately stop before Chroma and provider calls.  Provider and
embedding boundaries are covered by the opt-in smoke tests; this module keeps
path handling deterministic and cheap.
"""

from pathlib import Path, PureWindowsPath

import pytest

from akasha.RAG.rag import RAG
from akasha.utils.atman import atman
from akasha.utils.db.db_structure import dbs


pytestmark = pytest.mark.unit


def _uninitialised_rag() -> RAG:
    """Build only the object state needed by ``atman._get_db``."""
    rag = RAG.__new__(RAG)
    rag.db = None
    rag.ignored_files = []
    rag.use_chroma = False
    rag.embeddings_obj = object()
    rag.chunk_size = 1000
    rag.verbose = False
    rag.env_file = ""
    return rag


@pytest.mark.parametrize(
    "data_source",
    [
        Path("tests/data/rag/single_fact.txt"),
        Path("tests/data/rag/directory"),
        PureWindowsPath(r"C:\Users\today\Projects\akasha-update\tests\data\rag\single_fact.txt"),
    ],
)
def test_rag_db_loader_receives_file_directory_and_windows_paths(monkeypatch, data_source):
    rag = _uninitialised_rag()
    received = {}

    def fake_process_db(**kwargs):
        received.update(kwargs)
        return dbs(), ["ignored.txt"]

    monkeypatch.setattr("akasha.utils.atman.process_db", fake_process_db)

    atman._get_db(rag, data_source)

    assert received["data_source"] == data_source
    assert received["embeddings"] is rag.embeddings_obj
    assert received["chunk_size"] == rag.chunk_size
    assert rag.ignored_files == ["ignored.txt"]
    assert isinstance(rag.db, dbs)


def test_rag_check_doc_path_preserves_path_objects():
    rag = RAG.__new__(RAG)
    source = Path("tests/data/rag/single_fact.txt")

    assert RAG._check_doc_path(rag, source) is source


def test_rag_check_doc_path_uses_db_object_sentinel():
    rag = RAG.__new__(RAG)
    source = dbs()

    assert RAG._check_doc_path(rag, source) == "use dbs object"


def test_rag_check_db_rejects_missing_database():
    rag = RAG.__new__(RAG)
    rag.db = None

    with pytest.raises(OSError, match="document path not exist"):
        RAG._check_db(rag)

