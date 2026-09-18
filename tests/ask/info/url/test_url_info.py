"""Live contract test for ``ask(..., info=[url, url])``."""

import pytest

import akasha
from tests.support.live import load_test_env, require_keys


pytestmark = [
    pytest.mark.live,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


def test_gemini_ask_with_two_url_info_items():
    """The public callable API accepts two URL references in ``info``."""
    require_keys("GEMINI_API_KEY")
    env_file = load_test_env()

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

