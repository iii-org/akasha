"""MCP discovery -> tool invocation -> real agent event/log contracts."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.support.live import load_test_env, require_keys, require_ollama
from tests.support.paths import FIXTURES_ROOT, REPO_ROOT

import akasha


MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"
SERVER = FIXTURES_ROOT / "mcp" / "echo_server.py"

REQUIRED_KEYS = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}

_manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
MODEL_CASES = [
    pytest.param(
        item["provider"],
        item["id"],
        REQUIRED_KEYS.get(item["provider"]),
        id=item["id"],
    )
    for item in _manifest["models"]
]

def _load_test_env(required_key: str | None = None) -> None:
    if required_key:
        require_keys(required_key)
    else:
        load_test_env()


def _ollama_is_available() -> bool:
    require_ollama()
    return True


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "akasha_test": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER)],
            }
        }
    )


async def _discover_tools():
    client = _client()
    tools = await client.get_tools()
    return client, {tool.name: tool for tool in tools}


@pytest.mark.integration
@pytest.mark.contract
def test_mcp_discovery_and_direct_invocation():
    """The local stdio server exposes stable tools and callable schemas."""
    _load_test_env()
    _client_obj, tools = asyncio.run(_discover_tools())

    assert set(tools) == {"mcp_add", "mcp_get_weather", "mcp_lookup_version"}
    schema = tools["mcp_add"].args_schema
    schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"
    add_result = asyncio.run(tools["mcp_add"].ainvoke({"a": 20, "b": 22}))
    version_result = asyncio.run(
        tools["mcp_lookup_version"].ainvoke({"package": "akasha"})
    )
    assert add_result[0]["type"] == "text"
    assert add_result[0]["text"] == "42"
    assert version_result[0]["text"] == "akasha: MCP_TEST_VERSION_1.2.3"


@pytest.mark.parametrize("provider,model,required_key", MODEL_CASES)
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_mcp_tools_are_executed_by_real_agent_non_stream(
    provider, model, required_key
):
    """Every manifest model must select MCP and record its tool call."""
    _load_test_env(required_key)
    if provider == "ollama" and not _ollama_is_available():
        pytest.skip("configured Ollama endpoint is unavailable")
    _client_obj, tools = asyncio.run(_discover_tools())
    agent = akasha.agents(
        model=model,
        tools=list(tools.values()),
        stream=False,
        thinking=False,
        keep_logs=True,
        max_output_tokens=128,
        max_round=3,
    )

    response = agent(
        "You must use the mcp_add tool. Add 20 and 22, then reply with the result."
    )

    assert isinstance(response, str) and response.strip(), f"{provider} returned no answer"
    assert "42" in response, f"{provider} did not use the MCP result: {response!r}"
    assert any(call.get("name") == "mcp_add" for call in agent.tool_calls)
    json.dumps(agent.logs, ensure_ascii=False)


@pytest.mark.integration
@pytest.mark.contract
def test_mcp_tools_require_non_stream_mode():
    """MCP agents use ainvoke and reject the sync stream facade explicitly."""
    _load_test_env("OPENAI_API_KEY")
    _client_obj, tools = asyncio.run(_discover_tools())
    agent = akasha.agents(
        model=next(item["id"] for item in _manifest["models"] if item["provider"] == "openai"),
        tools=list(tools.values()),
        stream=True,
        thinking=False,
        keep_logs=True,
        max_output_tokens=128,
        max_round=3,
    )

    with pytest.raises(ValueError, match="async-only.*stream=False"):
        agent("Use mcp_add to add 7 and 8.")
