import pytest

from akasha.utils import upload
from akasha.utils.optional_dependencies import OptionalDependencyError

pytestmark = pytest.mark.unit


def test_mlflow_init_fails_cleanly_by_default_when_tracking_is_missing(monkeypatch):
    monkeypatch.setattr(
        upload,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("MLflow experiment tracking", "tracking")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        upload.mlflow_init()

    assert exc_info.value.extra == "tracking"


def test_mlflow_init_can_still_be_used_as_a_soft_probe(monkeypatch):
    monkeypatch.setattr(
        upload,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("MLflow experiment tracking", "tracking")
        ),
    )

    assert upload.mlflow_init(do_not_raise=True) is None
