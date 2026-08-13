"""Deterministic Streamable HTTP MCP server used by contract tests."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "akasha-streamable-http-test-mcp",
    host="127.0.0.1",
    port=int(os.environ.get("MCP_TEST_PORT", "8765")),
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


@mcp.tool()
def mcp_structured_add(a: int, b: int) -> dict[str, int]:
    """Add two integers and return a structured result."""
    return {"sum": a + b}


@mcp.tool()
def mcp_get_status() -> str:
    """Return a deterministic status string."""
    return "MCP_STREAMABLE_HTTP_OK"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
