"""Provider adapters are imported only at the selected runtime seam."""

import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def _run_probe(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_agent_import_does_not_load_provider_adapters():
    output = _run_probe(
        "import sys; "
        "import akasha.agent.agents; "
        "assert 'langchain_openai' not in sys.modules; "
        "assert 'langchain_google_genai' not in sys.modules; "
        "assert 'langchain_ollama' not in sys.modules"
    )
    assert output == ""


def test_selected_gemini_adapter_does_not_load_openai_sdk():
    output = _run_probe(
        "import sys; "
        "from akasha.utils.models.chat import build_chat_model; "
        "build_chat_model('gemini', 'gemini-2.5-flash', {'GEMINI_API_KEY': 'test'}); "
        "assert 'langchain_google_genai' in sys.modules; "
        "assert 'langchain_openai' not in sys.modules"
    )
    assert output == ""
