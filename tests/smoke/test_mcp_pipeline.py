"""MCP discovery -> tool invocation -> real agent event/log contracts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values, load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

import akasha


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / "tests" / ".env"
MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"
SERVER = REPO_ROOT / "tests" / "fixtures" / "mcp" / "echo_server.py"
RUN_LIVE = os.getenv("RUN_MCP_SMOKE", "").lower() in {"1", "true", "yes"}

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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_MCP_SMOKE=1 to enable MCP live agent smoke tests",
    ),
]


def _load_test_env(required_key: str | None = None) -> None:
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if required_key and not (os.getenv(required_key) or values.get(required_key)):
        pytest.skip(f"{required_key} is not configured")
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)


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
def test_mcp_tools_are_executed_by_real_agent_non_stream(
    provider, model, required_key
):
    """Every manifest model must select MCP and record its tool call."""
    _load_test_env(required_key)
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


def test_mcp_agent_requires_non_stream_mode():
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
