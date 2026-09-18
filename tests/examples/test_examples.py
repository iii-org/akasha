"""Examples must use current public signatures without contacting providers."""
import ast
import inspect
from pathlib import Path

import pytest
import akasha
import akasha.helper as ah
from akasha.utils.search.search_doc import retri_docs

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("filename,method,target", [
    ("ex_selfask_rag.py", "selfask_RAG", akasha.RAG.selfask_RAG),
    ("helper/ex_retriver.py", "retri_docs", retri_docs),
    ("helper/ex_token_count.py", "get_doc_length", ah.get_doc_length),
])
def test_example_calls_match_current_public_signature(filename, method, target):
    tree = ast.parse((ROOT / "examples" / filename).read_text(encoding="utf-8-sig"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name != method:
            continue
        found = True
        args = [None] * len(node.args)
        if method == "selfask_RAG":
            args.insert(0, None)
        inspect.signature(target).bind(*args, **{kw.arg: None for kw in node.keywords})
    assert found, f"{filename} no longer demonstrates {method}"

import importlib
import os
import subprocess
import sys

EXAMPLES = ROOT / "examples"
ENTRYPOINTS = sorted(
    list(EXAMPLES.glob("ex_*.py"))
    + list((EXAMPLES / "helper").glob("ex_*.py"))
    + list((EXAMPLES / "examples_skills").glob("*app*.py"))
    + list((EXAMPLES / "examples_skills").glob("app_*.py"))
    + [EXAMPLES / "mcp_server.py"]
)
ENTRYPOINTS = list(dict.fromkeys(ENTRYPOINTS))

@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_every_entrypoint_has_help_from_an_unrelated_directory(path, tmp_path):
    result = subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout

@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_importing_examples_does_not_call_services(path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Importing examples must not run workloads")
    for name in ("ask", "RAG", "summary", "websearch", "agents", "MemoryManager", "eval", "gen_image", "edit_image"):
        monkeypatch.setattr(akasha, name, forbidden)
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    module_name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    importlib.import_module(module_name)

@pytest.fixture
def offline_embeddings(monkeypatch):
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.embeddings import DeterministicFakeEmbedding
    fake = DeterministicFakeEmbedding(size=16)
    monkeypatch.setenv("OPENAI_API_KEY", "example-offline-test")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr(OpenAIEmbeddings, "embed_documents", lambda self, texts, **kwargs: fake.embed_documents(texts))
    monkeypatch.setattr(OpenAIEmbeddings, "embed_query", lambda self, text, **kwargs: fake.embed_query(text))

@pytest.mark.parametrize("module_name", [
    "examples.helper.ex_load_db", "examples.helper.ex_remove_db", "examples.helper.ex_retriver",
])
def test_database_examples_run_with_real_chroma_and_offline_embeddings(module_name, tmp_path, offline_embeddings, capsys):
    module = importlib.import_module(module_name)
    previous = Path.cwd()
    module.main(["--output-dir", str(tmp_path)])
    assert Path.cwd() == previous
    assert "Artifacts:" in capsys.readouterr().out
    assert list(tmp_path.glob("*/chromadb"))

def test_api_example_payloads_match_server_models(monkeypatch):
    from argparse import Namespace
    from examples.ex_api import build_payload
    from akasha.api import InfoModel, ConsultModel, SummaryModel, webInfoModel
    monkeypatch.setenv("OPENAI_API_KEY", "example-offline-test")
    for action, schema in [("ask", InfoModel), ("rag", ConsultModel),
                           ("summary", SummaryModel), ("websearch", webInfoModel)]:
        args = Namespace(action=action, model="openai:gpt-4o-mini",
                         embeddings="openai:text-embedding-3-small",
                         data_source=None, engine="wiki")
        payload = build_payload(args)
        assert not set(payload) - set(schema.model_fields)
        parsed = schema.model_validate(payload)
        assert parsed.env_config["OPENAI_API_KEY"] == "example-offline-test"
    assert build_payload(Namespace(action="summary", model="openai:gpt-4o-mini"))["summary_type"] == "map_reduce"

def test_stream_renderer_handles_string_and_event_contracts(capsys):
    from examples._common import print_response
    assert print_response(iter(["hello", " world"])) == "hello world"
    assert print_response(iter([{"type": "progress", "data": "checking"},
                                {"type": "answer", "data": "done"}])) == "done"
    assert "[progress] checking" in capsys.readouterr().out


def test_self_query_example_can_reload_persisted_metadata(tmp_path, offline_embeddings, monkeypatch, capsys):
    from examples.helper.ex_self_query import main
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    replies = iter([
        '{"query": "maintenance expenses", "filter": "and(eq(\\"factory\\", \\"A\\"), eq(\\"year\\", 2024))"}',
        "Factory A spent 1200 dollars in 2024.",
    ])
    def generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=next(replies)))])
    monkeypatch.setattr(ChatOpenAI, "_generate", generate)
    main(["--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "factory_a_2024.txt" in output
    assert "factory_b_2024.txt" not in output
    assert "'year': 2024" in output


def test_bundled_mcp_server_discovers_and_executes_tools():
    import asyncio
    from langchain_mcp_adapters.client import MultiServerMCPClient
    async def exercise():
        client = MultiServerMCPClient({"example": {
            "transport": "stdio", "command": sys.executable,
            "args": [str(EXAMPLES / "mcp_server.py"), "--transport", "stdio"],
        }})
        tools = {tool.name: tool for tool in akasha.normalize_mcp_tools(await client.get_tools())}
        assert set(tools) == {"add", "get_weather"}
        result = await tools["add"].ainvoke({"a": 20, "b": 22})
        assert result == 42 or result == {"result": 42}
    asyncio.run(exercise())


@pytest.mark.parametrize("follow_up", [[], ["What do sensors measure?"]])
def test_selfask_example_streams_final_text_and_saves_completed_logs(
    tmp_path, offline_embeddings, monkeypatch, capsys, follow_up,
):
    import json
    from examples.ex_selfask_rag import main
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    replies = iter([json.dumps({"need": bool(follow_up), "follow_up": follow_up}),
                    "Sensors measure vibration."])
    observed = []
    def generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=next(replies)))])
    def stream(self, messages, **kwargs):
        prompt = str(messages)
        observed.append(prompt)
        assert "generator object" not in prompt
        if follow_up:
            assert "Sensors measure vibration." in prompt
        for text in ["Final ", "answer."]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
    monkeypatch.setattr(ChatOpenAI, "_generate", generate)
    monkeypatch.setattr(ChatOpenAI, "_stream", stream)
    main(["--stream", "--output-dir", str(tmp_path)])
    assert len(observed) == 1
    assert "Final answer." in capsys.readouterr().out
    log_file = next(tmp_path.glob("*/selfask.json"))
    logs = json.loads(log_file.read_text(encoding="utf-8"))
    assert any(item.get("response") == "Final answer." for item in logs.values())


def test_websearch_uses_configured_engine_without_requiring_wikipedia(tmp_path, monkeypatch, capsys):
    from examples.ex_websearch import main
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from importlib import import_module
    web_module = import_module("akasha.tools.websearch")
    monkeypatch.setenv("OPENAI_API_KEY", "example-offline-test")
    monkeypatch.setenv("AKASHA_SEARCH_ENGINE", "brave")
    seen = []
    def search(prompt, search_engine="wiki", search_num=5, language="ch", env_file=""):
        seen.append(search_engine)
        return [Document(page_content="Industry 4.0 connects industrial equipment.")]
    monkeypatch.setattr(web_module, "load_docs_from_webengine", search)
    monkeypatch.setattr(ChatOpenAI, "_generate", lambda self, messages, **kwargs:
                        ChatResult(generations=[ChatGeneration(message=AIMessage(content="Connected manufacturing."))]))
    main(["--output-dir", str(tmp_path)])
    assert seen == ["brave"]
    assert "Connected manufacturing." in capsys.readouterr().out
