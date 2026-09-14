"""Contract tests for the current Streamable HTTP MCP transport."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import akasha
from langchain_mcp_adapters.client import MultiServerMCPClient
from tests.support.paths import FIXTURES_ROOT, REPO_ROOT

SERVER = FIXTURES_ROOT / "mcp" / "streamable_http_server.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["MCP_TEST_PORT"] = str(port)
    return subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_server(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"MCP server exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("MCP Streamable HTTP server did not start")


def test_streamable_http_discovery_and_direct_invocation():
    port = _free_port()
    process = _start_server(port)
    try:
        _wait_for_server(port, process)
        client = MultiServerMCPClient(
            {
                "streamable_test": {
                    "transport": "streamable_http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                }
            }
        )

        async def exercise():
            discovered = await client.get_tools()
            tools = {tool.name: tool for tool in akasha.normalize_mcp_tools(discovered)}
            structured = await tools["mcp_structured_add"].ainvoke({"a": 20, "b": 22})
            status = await tools["mcp_get_status"].ainvoke({})
            return tools, structured, status

        tools, structured, status = asyncio.run(exercise())
        assert set(tools) == {"mcp_structured_add", "mcp_get_status"}
        assert structured["sum"] == 42
        assert status == "MCP_STREAMABLE_HTTP_OK"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
