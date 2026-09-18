import pytest

MODEL_ID = "ModelCloud/Qwen2.5-0.5B-Instruct-gptqmodel-w4a16"
MODEL_REVISION = "7911278f62450ff588b7f1bb6c0778fba839ede8"

pytestmark = [
    pytest.mark.live,
    pytest.mark.full_only,
    pytest.mark.contract,
]


def test_gptqmodel_loads_and_generates_on_cpu():
    """The maintained GPTQ backend must complete real CPU inference."""

    for dependency in ("torch", "transformers", "gptqmodel"):
        try:
            __import__(dependency)
        except ImportError as exc:
            pytest.fail(f"full installation is missing {dependency}: {exc}")

    from akasha.utils.models.gtq import gptq

    model = gptq(
        MODEL_ID,
        device="cpu",
        revision=MODEL_REVISION,
        temperature=0.01,
        max_token=8,
    )
    response = model.invoke("The capital of France is")

    assert model.quant_backend == "gptqmodel"
    assert model.quant_device == "cpu"
    assert isinstance(response, str)
    assert response.strip()
