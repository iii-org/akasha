import pytest
from langchain_core.documents import Document

from akasha.tools.ask import ask
import akasha.helper.token_counter as token_counter


pytestmark = pytest.mark.unit


def test_separate_docs_drops_empty_chunks(monkeypatch):
    monkeypatch.setattr(
        token_counter.myTokenizer,
        "compute_tokens",
        lambda text, model: len(text.split()),
    )

    qa = object.__new__(ask)
    qa.max_input_tokens = 5
    qa.prompt_tokens = 1
    qa.model = "test:model"
    qa.docs = [
        Document(page_content="one two three four five six seven eight"),
        Document(page_content="nine ten eleven twelve thirteen fourteen fifteen"),
    ]

    chunks, total_tokens = qa._separate_docs()

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert total_tokens == 15

