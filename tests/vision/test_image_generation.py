from types import SimpleNamespace
from PIL import Image

import akasha.tools.gen_img as gen_img_module
import pytest

from akasha.utils.models.gemi import gemini_model


pytestmark = pytest.mark.unit


class FakeImageClient:
    def __init__(self):
        self.generate_calls = []
        self.edit_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self._save_image(kwargs["save_path"])

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return self._save_image(kwargs["save_path"])


    @staticmethod
    def _save_image(save_path):
        with Image.new("RGB", (2, 2), "white") as image:
            image.save(save_path, format="PNG")
        return str(save_path)


class FakeGeminiModels:
    def __init__(self, response):
        self.response = response
        self.generate_calls = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.response


def test_gemini_generate_raises_clear_error_for_no_image_response(tmp_path):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="No image generated", inline_data=None)]
                ),
                finish_reason="NO_IMAGE",
            )
        ],
        prompt_feedback=SimpleNamespace(block_reason="IMAGE_SAFETY"),
    )
    client = gemini_model(
        model_name="gemini-3.1-flash-image",
        api_key="unused",
    )
    client.client = SimpleNamespace(models=FakeGeminiModels(response))

    with pytest.raises(RuntimeError) as exc_info:
        client.generate(
            prompt="a long-tailed tit on a branch",
            save_path=str(tmp_path / "generated.png"),
            verbose=False,
        )

    message = str(exc_info.value)
    assert "finish_reason=NO_IMAGE" in message
    assert "prompt_feedback.block_reason=IMAGE_SAFETY" in message


def test_gemini_generate_explicitly_disables_afc(tmp_path):
    response = SimpleNamespace(candidates=[], prompt_feedback=None)
    models = FakeGeminiModels(response)
    client = gemini_model(
        model_name="gemini-3.1-flash-image",
        api_key="unused",
    )
    client.client = SimpleNamespace(models=models)

    with pytest.raises(RuntimeError):
        client.generate(
            prompt="a long-tailed tit on a branch",
            save_path=str(tmp_path / "generated.png"),
            verbose=False,
        )

    config = models.generate_calls[0]["config"]
    assert config.automatic_function_calling.disable is True


def test_gemini_edit_raises_for_no_image_and_disables_afc(tmp_path):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[]),
                finish_reason="NO_IMAGE",
            )
        ],
        prompt_feedback=SimpleNamespace(block_reason="IMAGE_SAFETY"),
    )
    models = FakeGeminiModels(response)
    client = gemini_model(
        model_name="gemini-3.1-flash-image",
        api_key="unused",
    )
    client.client = SimpleNamespace(
        models=models,
        files=SimpleNamespace(upload=lambda file: "uploaded-image"),
    )
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"image fixture")

    with pytest.raises(RuntimeError) as exc_info:
        client.edit(
            prompt="add a second bird",
            images=[source_path],
            save_path=str(tmp_path / "edited.png"),
            verbose=False,
        )

    message = str(exc_info.value)
    assert "finish_reason=NO_IMAGE" in message
    assert "prompt_feedback.block_reason=IMAGE_SAFETY" in message
    config = models.generate_calls[0]["config"]
    assert config.automatic_function_calling.disable is True


def test_gen_image_delegates_generation_options(monkeypatch, tmp_path, capsys, previewed_images):
    client = FakeImageClient()
    monkeypatch.setattr(gen_img_module, "handle_client", lambda model, env_file: client)

    save_path = tmp_path / "generated.png"
    result = gen_img_module.gen_image(
        prompt="a red bicycle",
        save_path=str(save_path),
        model="openai:gpt-image-1",
        size="1024x1024",
        quality="high",
        verbose=True,
        env_file="tests/.env",
    )

    assert "Failed to open image" not in capsys.readouterr().out
    assert previewed_images == [(str(save_path), (2, 2))]
    assert result == str(save_path)
    assert client.generate_calls == [
        {
            "prompt": "a red bicycle",
            "save_path": str(save_path),
            "size": "1024x1024",
            "quality": "high",
            "moderation": "auto",
            "background": "auto",
            "verbose": True,
        }
    ]


def test_edit_image_normalizes_single_path_and_delegates(monkeypatch, tmp_path, capsys, previewed_images):
    client = FakeImageClient()
    monkeypatch.setattr(gen_img_module, "handle_client", lambda model, env_file: client)

    source_path = tmp_path / "source.png"
    save_path = tmp_path / "edited.png"
    source_path.write_bytes(b"image fixture")

    result = gen_img_module.edit_image(
        prompt="remove the bicycle",
        images=source_path,
        save_path=str(save_path),
        model="openai:gpt-image-1",
        size="1024x1024",
        quality="medium",
        verbose=True,
    )

    assert "Failed to open image" not in capsys.readouterr().out
    assert previewed_images == [(str(save_path), (2, 2))]
    assert result == str(save_path)
    assert client.edit_calls == [
        {
            "prompt": "remove the bicycle",
            "images": [source_path],
            "save_path": str(save_path),
            "size": "1024x1024",
            "quality": "medium",
            "moderation": "auto",
            "background": "auto",
            "verbose": True,
        }
    ]


def test_edit_image_preserves_multiple_sources(monkeypatch, tmp_path):
    client = FakeImageClient()
    monkeypatch.setattr(gen_img_module, "handle_client", lambda model, env_file: client)

    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    result = gen_img_module.edit_image(
        prompt="combine these references",
        images=sources,
        save_path=str(tmp_path / "combined.png"),
    )

    assert result == str(tmp_path / "combined.png")
    assert client.edit_calls[0]["images"] == sources


@pytest.fixture
def previewed_images(monkeypatch):
    """Keep real PNG decoding, but do not launch a desktop viewer in CI."""
    shown = []

    def record_preview(image, *args, **kwargs):
        image.load()
        shown.append((str(image.filename), image.size))

    monkeypatch.setattr(Image.Image, "show", record_preview)
    return shown
