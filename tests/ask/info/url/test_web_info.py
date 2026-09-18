"""Opt-in live test for asking about Akasha from web-provided context.

The test intentionally calls the real Gemini API and fetches the URLs in
``info``.  It is disabled by default so ordinary test runs do not require
network access or spend provider quota.
"""

import pytest

import akasha
from tests.support.live import load_test_env, require_keys


pytestmark = [
    pytest.mark.live,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


def _env_file() -> str:
    return load_test_env()


def _require_gemini_key() -> None:
    require_keys("GEMINI_API_KEY")


def test_gemini_ask_answers_from_web_info():
    """Gemini should identify Akasha as a flexible LLM QA/RAG tool."""
    _require_gemini_key()

    qa = akasha.ask(
        model="gemini:gemini-3.5-flash",
        max_output_tokens=65536,
        env_file=_env_file(),
    )
    info = [
        "https://pypi.org/project/akasha-terminal/",
        "[iii-org/akasha](https://github.com/iii-org/akasha)",
    ]

    response = qa(
        "What is akasha-terminal? Based only on the provided references, answer briefly about the package and its document QA purpose.",
        info=info,
    )

    assert isinstance(response, str)
    assert response.strip()

    normalized = response.casefold()
    assert any(term in normalized for term in ("akasha", "akasha-terminal", "terminal"))
    assert any(term in normalized for term in ("langchain", "chromadb", "document", "package", "qa", "question"))

    print(f"\n[Gemini web-info ask] Response: {response}")
