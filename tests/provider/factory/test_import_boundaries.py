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


def test_second_stage_reranker_does_not_load_local_model_stack():
    output = _run_probe(
        "import sys; "
        "from akasha.utils.search.rerank import rerank_documents; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert callable(rerank_documents)"
    )
    assert output == ""


def test_rag_import_does_not_require_optional_feature_stacks():
    output = _run_probe(
        "import builtins, sys; "
        "original_import = builtins.__import__; "
        "blocked = {'faiss', 'mlflow', 'streamlit', 'unstructured'}; "
        "builtins.__import__ = lambda name, *args, **kwargs: "
        "(_ for _ in ()).throw(ImportError(name)) "
        "if name.split('.', 1)[0] in blocked "
        "else original_import(name, *args, **kwargs); "
        "from akasha.RAG.rag import RAG; "
        "assert RAG is not None; "
        "assert 'torch' not in sys.modules"
    )
    assert output == ""


def test_api_import_and_cleanup_do_not_require_torch():
    output = _run_probe(
        "import builtins, sys; "
        "original_import = builtins.__import__; "
        "builtins.__import__ = lambda name, *args, **kwargs: "
        "(_ for _ in ()).throw(ImportError(name)) "
        "if name.split('.', 1)[0] == 'torch' "
        "else original_import(name, *args, **kwargs); "
        "import akasha.api; "
        "akasha.api.clean(); "
        "assert 'torch' not in sys.modules"
    )
    assert output == ""
