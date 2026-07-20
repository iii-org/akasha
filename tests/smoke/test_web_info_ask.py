"""Opt-in live test for asking about Akasha from web-provided context.

The test intentionally calls the real Gemini API and fetches the URLs in
``info``.  It is disabled by default so ordinary test runs do not require
network access or spend provider quota.
"""

import os
from pathlib import Path

import pytest

import akasha


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes"},
        reason="set RUN_LLM_TESTS=1 to enable live API tests",
    ),
]


def _env_file() -> str:
    configured = os.getenv("ENV_FILE")
    if configured:
        return configured

    root_env = Path(__file__).resolve().parents[2] / ".env"
    return str(root_env) if root_env.exists() else ".env"


def _require_gemini_key() -> None:
    if not os.getenv("GEMINI_API_KEY") and not Path(_env_file()).exists():
        pytest.skip("GEMINI_API_KEY or a readable ENV_FILE is required")


def test_gemini_ask_answers_from_web_info():
    """Gemini should identify Akasha as a flexible LLM QA/RAG tool."""
    _require_gemini_key()

    qa = akasha.ask(
        model="gemini:gemini-3.5-flash",
        max_output_tokens=256,
        env_file=_env_file(),
    )
    info = [
        "https://pypi.org/project/akasha-terminal/",
        "[iii-org/akasha](https://github.com/iii-org/akasha)",
    ]

    response = qa("akasha 是做什麼的?", info=info)

    assert isinstance(response, str)
    assert response.strip()

    normalized = response.casefold()
    assert any(term in normalized for term in ("akasha-terminal", "akasha terminal"))
    assert "rag" in normalized
    assert any(
        term in normalized
        for term in (
            "大語言模型",
            "大型語言模型",
            "language model",
            "llm",
        )
    )
    assert any(
        term in normalized
        for term in ("問答", "question answering", "qa")
    )
    assert any(term in normalized for term in ("多種模型", "各種模型", "模型"))
    assert any(term in normalized for term in ("彈性", "靈活", "flexible"))

    print(f"\n[Gemini web-info ask] Response: {response}")
