import pytest
from langchain_core.documents import Document

from akasha.utils.db import file_loader, load_docs
from akasha.utils.optional_dependencies import OptionalDependencyError

pytestmark = pytest.mark.unit


def test_missing_pptx_stack_points_to_documents_extra(monkeypatch):
    monkeypatch.setattr(
        file_loader,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("PPTX loading", "documents")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        file_loader.load_file("example.pptx", "pptx")

    assert exc_info.value.extra == "documents"
    assert 'uv add "akasha-terminal[documents]"' in str(exc_info.value)
    assert 'pip install "akasha-terminal[documents]"' in str(exc_info.value)


def test_directory_skips_optional_document_when_other_documents_load(monkeypatch):
    monkeypatch.setattr(
        file_loader,
        "get_load_file_list",
        lambda _path, extension: (
            ["slides.pptx"]
            if extension == "pptx"
            else ["notes.txt"]
            if extension == "txt"
            else []
        ),
    )

    def fake_load_file(path, extension):
        if extension == "pptx":
            raise OptionalDependencyError("PPTX loading", "documents")
        return [Document(page_content="notes", metadata={"source": str(path)})]

    monkeypatch.setattr(file_loader, "load_file", fake_load_file)

    with pytest.warns(
        RuntimeWarning, match=r"slides\.pptx.*akasha-terminal\[documents\]"
    ):
        docs = file_loader.load_directory("example")

    assert [doc.page_content for doc in docs] == ["notes"]


def test_directory_raises_optional_error_when_nothing_can_be_loaded(monkeypatch):
    monkeypatch.setattr(
        file_loader,
        "get_load_file_list",
        lambda _path, extension: ["slides.pptx"] if extension == "pptx" else [],
    )
    monkeypatch.setattr(
        file_loader,
        "load_file",
        lambda *_args: (_ for _ in ()).throw(
            OptionalDependencyError("PPTX loading", "documents")
        ),
    )

    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(OptionalDependencyError, match=r"akasha-terminal\[documents\]"),
    ):
        file_loader.load_directory("example")


def test_direct_file_loading_does_not_turn_optional_error_into_text(
    monkeypatch, tmp_path
):
    pptx_path = tmp_path / "slides.pptx"
    pptx_path.touch()
    monkeypatch.setattr(
        load_docs,
        "load_file",
        lambda *_args: (_ for _ in ()).throw(
            OptionalDependencyError("PPTX loading", "documents")
        ),
    )

    with pytest.raises(OptionalDependencyError):
        load_docs.load_docs_from_info(pptx_path)
