"""Live smoke test for the public Ask callable."""

import pytest

import akasha
from tests.support.live import load_test_env, require_keys


pytestmark = [
    pytest.mark.live,
    pytest.mark.contract,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


def test_gemini_ask_returns_visible_text_and_serializable_logs():
    require_keys("GEMINI_API_KEY")
    qa = akasha.ask(
        model="gemini:gemini-2.5-flash",
        keep_logs=True,
        max_output_tokens=64,
        env_file=load_test_env(),
    )

    response = qa("Reply with exactly: ASK_SMOKE_OK")

    assert isinstance(response, str)
    assert response.strip()
    assert qa.response == response
    assert qa.logs
