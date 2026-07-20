"""Deterministic stdio MCP server used by the MCP smoke tests."""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("akasha-test-mcp")


@mcp.tool()
def mcp_add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def mcp_get_weather(city: str) -> str:
    """Return deterministic weather for a test city."""
    return f"{city}: MCP_TEST_WEATHER_SUNNY"


@mcp.tool()
def mcp_lookup_version(package: str) -> str:
    """Return a deterministic package version for contract testing."""
    return f"{package}: MCP_TEST_VERSION_1.2.3"


if __name__ == "__main__":
    mcp.run(transport="stdio")
