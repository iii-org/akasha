"""Live contract test for ``ask(..., info=[url, url])``."""

import os
from pathlib import Path

import pytest

import akasha
from tests.support.paths import TEST_ENV_FILE


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes"},
        reason="set RUN_LLM_TESTS=1 to enable live API tests",
    ),
]


def test_gemini_ask_with_two_url_info_items():
    """The public callable API accepts two URL references in ``info``."""
    env_file = os.getenv("ENV_FILE", str(TEST_ENV_FILE))
    if not os.getenv("GEMINI_API_KEY") and not Path(env_file).exists():
        pytest.skip("GEMINI_API_KEY or a readable ENV_FILE is required")

    qa = akasha.ask("gemini:gemini-2.5-flash", 
                    env_file=env_file)
    answer = qa(
        "akasha-terminal\u662f\u4ec0\u9ebc?",
        max_input_tokens=1048576,
        max_output_tokens=65536,
        info=[
            "https://github.com/iii-org/akasha",
            "https://pypi.org/project/akasha-terminal/",
        ],
    )
    print(f"Answer: {answer}")
    assert isinstance(answer, str)
    assert answer.strip()

