from pathlib import Path

import akasha
import pytest

from tests.support.live import load_test_env, require_keys


VISION_CASES = [
    pytest.param(
        "gemini:gemini-2.5-flash",
        "gemini:gemini-3.1-flash-image",
        "GEMINI_API_KEY",
        id="gemini",
    ),
    pytest.param(
        "openai:gpt-5.6-luna",
        "openai:gpt-image-2-2026-04-21",
        "OPENAI_API_KEY",
        id="openai",
    ),
]


pytestmark = [
    pytest.mark.live,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


@pytest.mark.parametrize(
    "understanding_model,image_model,required_key",
    VISION_CASES,
)
def test_live_vision_generate_understand_and_edit(
    tmp_path: Path,
    understanding_model: str,
    image_model: str,
    required_key: str,
):
    """Verify each provider's real image generation, vision, and editing path."""
    require_keys(required_key)
    env_file = load_test_env()
    generated_path = tmp_path / "generated.png"
    edited_path = tmp_path / "edited.png"

    generated = akasha.gen_image(
        prompt="在樹上枝枒上有一隻長尾山雀",
        model=image_model,
        save_path=str(generated_path),
        env_file=env_file,
    )
    assert Path(generated).exists()

    asker = akasha.ask(
        model=understanding_model,
        env_file=env_file,
    )
    answer = asker.vision(
        prompt="這張圖片有什麼東西？請描述主要的動物與場景。",
        image_path=str(generated_path),
    )
    assert isinstance(answer, str)
    assert answer.strip()

    edited = akasha.edit_image(
        prompt="再加一隻長尾山雀在旁邊",
        images=generated_path,
        model=image_model,
        save_path=str(edited_path),
        env_file=env_file,
    )
    assert Path(edited).exists()
