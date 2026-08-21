import akasha.tools.gen_img as gen_img_module


class FakeImageClient:
    def __init__(self):
        self.generate_calls = []
        self.edit_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return "generated.png"

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return "edited.png"


def test_gen_image_delegates_generation_options(monkeypatch, tmp_path):
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

    assert result == "generated.png"
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


def test_edit_image_normalizes_single_path_and_delegates(monkeypatch, tmp_path):
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

    assert result == "edited.png"
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

    assert result == "edited.png"
    assert client.edit_calls[0]["images"] == sources
