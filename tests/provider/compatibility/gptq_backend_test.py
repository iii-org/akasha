import sys
from types import ModuleType

import pytest

from akasha.utils.models import gtq as gtq_module
from akasha.utils.optional_dependencies import OptionalDependencyError

pytestmark = [pytest.mark.unit, pytest.mark.full_only]


def test_gptq_loader_falls_back_to_maintained_gptqmodel(monkeypatch):
    loaded = []
    sentinel = object()

    class FakeGPTQModel:
        @staticmethod
        def load(model_name):
            loaded.append(model_name)
            return sentinel

    replacement = ModuleType("gptqmodel")
    replacement.GPTQModel = FakeGPTQModel
    monkeypatch.setitem(sys.modules, "auto_gptq", None)
    monkeypatch.setitem(sys.modules, "gptqmodel", replacement)

    model, backend = gtq_module._load_quantized_model("org/model")

    assert model is sentinel
    assert backend == "gptqmodel"
    assert loaded == ["org/model"]


def test_peft_loader_reports_peft_extra_when_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(
        gtq_module,
        "require_optional_dependency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OptionalDependencyError("PEFT models", "peft")
        ),
    )

    with pytest.raises(OptionalDependencyError) as exc_info:
        gtq_module._get_peft_model_class()

    assert exc_info.value.extra == "peft"
