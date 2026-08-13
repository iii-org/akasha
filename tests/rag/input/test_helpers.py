import pytest
from langchain_core.documents import Document

from akasha.helper import base as helper_base

pytestmark = pytest.mark.unit


def test_separate_name_and_embedding_name_helpers():
    assert helper_base.separate_name("openai:text-embedding-3-small") == (
        "openai",
        "text-embedding-3-small",
    )
    assert helper_base.separate_name("hf:model:with:colon") == ("hf", "model:with:colon")
    assert helper_base.separate_name("plain") == ("plain", "")

    def fake_embed():
        return None

    assert helper_base.get_embedding_type_and_name("gemini:embedding-001") == (
        "gemini",
        "embedding-001",
    )
    assert helper_base.get_embedding_type_and_name(fake_embed) == ("fake_embed", "")


def test_doc_length_helpers_and_conversion(monkeypatch):
    class _FakeTokenizer:
        @staticmethod
        def cut_text(text):
            return ["中", "文", "測", "試"]

    monkeypatch.setattr(helper_base, "jieba_tokenizer", _FakeTokenizer())

    assert helper_base.get_doc_length("ch", "中文測試") == 4
    assert helper_base.get_doc_length("en", "one two three") == 3
    assert helper_base.get_docs_length("en", [Document(page_content="one two"), Document(page_content="three")]) == 3
    assert helper_base.sim_to_trad("简体中文")


def test_extract_json_and_extract_multiple_json():
    assert helper_base.extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}
    assert helper_base.extract_json("```json\n{'a': 1,}\n```") == {"a": 1}

    parsed = helper_base.extract_multiple_json("prefix {'a': '1'} middle {'b': False} suffix")
    assert parsed == [{"a": "1"}, {"b": False}]
