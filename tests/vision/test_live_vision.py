import os
from pathlib import Path

import akasha
import pytest

from tests.support.paths import TEST_ENV_FILE


RUN_VISION_TESTS = os.getenv("RUN_VISION_TESTS") == "1"


VISION_CASES = [
    pytest.param(
        "gemini:gemini-2.5-flash",
        "gemini:gemini-2.5-flash-image",
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
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_VISION_TESTS,
        reason="RUN_VISION_TESTS=1 is required for live vision tests",
    ),
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
    if not TEST_ENV_FILE.exists() and not os.getenv(required_key):
        pytest.skip(f"{required_key} or {TEST_ENV_FILE} is required")

    env_file = str(TEST_ENV_FILE) if TEST_ENV_FILE.exists() else ""
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
