"""Minimal MCP server using the current Streamable HTTP transport."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "akasha-example",
    host="127.0.0.1",
    port=int(os.getenv("MCP_PORT", "8000")),
    streamable_http_path="/mcp",
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def get_weather(city: str) -> str:
    """Return a small deterministic weather response for a city."""
    return f"{city}: sunny"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
