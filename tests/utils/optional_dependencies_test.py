import pytest

from akasha.utils import atman as atman_module
from akasha.utils.optional_dependencies import OptionalDependencyError

pytestmark = pytest.mark.unit


def test_optional_dependency_error_has_consistent_install_commands():
    error = OptionalDependencyError("MLflow experiment tracking", "tracking")

    assert error.feature == "MLflow experiment tracking"
    assert error.extra == "tracking"
    assert 'uv add "akasha-terminal[tracking]"' in str(error)
    assert 'pip install "akasha-terminal[tracking]"' in str(error)


def test_record_exp_is_validated_before_model_setup(monkeypatch):
    model_setup_calls = []
    monkeypatch.setattr(
        atman_module,
        "require_mlflow",
        lambda: (_ for _ in ()).throw(
            OptionalDependencyError("MLflow experiment tracking", "tracking")
        ),
    )
    monkeypatch.setattr(
        atman_module,
        "handle_model",
        lambda *_args, **_kwargs: model_setup_calls.append(True),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        atman_module.basic_llm(model=lambda prompt: prompt, record_exp="experiment")

    assert exc_info.value.extra == "tracking"
    assert model_setup_calls == []


def test_record_exp_override_is_validated_before_state_change(monkeypatch):
    monkeypatch.setattr(
        atman_module, "handle_model", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(atman_module, "handle_model_type", lambda value, *_args: value)
    client = atman_module.basic_llm(model=lambda prompt: prompt)
    monkeypatch.setattr(
        atman_module,
        "require_mlflow",
        lambda: (_ for _ in ()).throw(
            OptionalDependencyError("MLflow experiment tracking", "tracking")
        ),
    )

    with pytest.raises(OptionalDependencyError):
        client._change_variables(record_exp="experiment")

    assert client.record_exp == ""


def test_search_dependency_is_validated_before_model_setup(monkeypatch):
    from akasha.utils.search.retrievers import base as retriever_base

    model_setup_calls = []
    monkeypatch.setattr(
        retriever_base,
        "validate_search_type_dependencies",
        lambda _search_type: (_ for _ in ()).throw(
            OptionalDependencyError("FAISS retrieval", "faiss")
        ),
    )
    monkeypatch.setattr(
        atman_module,
        "handle_model",
        lambda *_args, **_kwargs: model_setup_calls.append(True),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        atman_module.atman(
            model=lambda prompt: prompt,
            embeddings=lambda texts: texts,
            search_type="faiss",
        )

    assert exc_info.value.extra == "faiss"
    assert model_setup_calls == []
