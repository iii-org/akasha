import pytest
from pathlib import PureWindowsPath
from langchain_core.documents import Document

from akasha.utils.db.db_structure import (
    NO_PARENT_DIR_NAME,
    _sanitize_path_part,
    dbs,
    get_storage_directory,
    is_url,
    _sanitize_path_string,
)

pytestmark = pytest.mark.unit


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeChroma:
    def __init__(self, payload):
        self.payload = payload

    def get(self, include=None):
        return self.payload


def test_dbs_defaults_to_empty_lists():
    store = dbs()

    assert store.get_ids() == []
    assert store.get_docs() == []
    assert store.get_metadatas() == []
    assert store.get_embeds() == []


def test_dbs_initializes_from_chroma_like_object_with_fallbacks():
    payload = {
        "ids": ["1", "2"],
        "embeddings": _FakeArray([[0.1], [0.2]]),
        "metadatas": None,
        "documents": ["doc-1", "doc-2"],
    }

    store = dbs(_FakeChroma(payload))

    assert store.get_ids() == ["1", "2"]
    assert store.get_embeds() == [[0.1], [0.2]]
    assert store.get_metadatas() == [{}, {}]
    assert store.get_docs() == ["doc-1", "doc-2"]


def test_merge_and_add_chromadb_keep_unique_ids():
    store = dbs()
    store.ids = ["1"]
    store.embeds = [[0.1]]
    store.metadatas = [{"source": "a"}]
    store.docs = ["alpha"]
    store.vis = {"1"}

    incoming = dbs()
    incoming.ids = ["1", "2"]
    incoming.embeds = [[0.1], [0.2]]
    incoming.metadatas = [{"source": "a"}, {"source": "b"}]
    incoming.docs = ["alpha", "beta"]
    incoming.vis = {"1", "2"}

    store.merge(incoming)

    assert store.get_ids() == ["1", "2"]
    assert store.get_docs() == ["alpha", "beta"]

    store.add_chromadb(
        _FakeChroma(
            {
                "ids": ["3"],
                "embeddings": [[0.3]],
                "metadatas": [{"source": "c"}],
                "documents": ["gamma"],
            }
        )
    )

    assert store.get_ids()[-1] == "3"
    assert store.get_docs()[-1] == "gamma"


def test_get_documents_returns_langchain_documents():
    store = dbs()
    store.docs = ["hello"]
    store.metadatas = [{"topic": "demo"}]

    docs = store.get_Documents()

    assert docs == [Document(page_content="hello", metadata={"topic": "demo"})]


def test_storage_directory_handles_path_dot_and_url():
    assert get_storage_directory(".", 100, "openai", "text-embedding-3-small") == (
        f"chromadb/{NO_PARENT_DIR_NAME}_openai_text-embedding-3-small_100"
    )
    assert get_storage_directory("docs/my_folder", 50, "hf", "bge/base") == (
        "chromadb/docs-myfolder_hf_bge-base_50"
    )

    url_dir = get_storage_directory("https://example.com/a/b?q=1", 25, "openai", "embed")
    assert url_dir.startswith("chromadb/httpsexamplecomabq1_openai_embed_25")


def test_storage_directory_sanitizes_windows_absolute_paths():
    storage_directory = get_storage_directory(
        PureWindowsPath(r"C:\Users\today\Projects\akasha-update\akasha-repo"),
        100,
        "openai",
        "text-embedding-3-small",
    )

    assert storage_directory == (
        "chromadb/C-Users-today-Projects-akasha-update-akasha-repo_"
        "openai_text-embedding-3-small_100"
    )
    assert ":" not in storage_directory
    assert "\\" not in storage_directory


def test_url_and_path_sanitization_helpers():
    assert is_url("https://example.com") is True
    assert is_url("ftp://example.com") is False
    assert _sanitize_path_string("https://exa mple.com/a-b_c?x=1") == "httpsexamplecomabcx1"
    assert _sanitize_path_part(r"C:\\") == "C"
