"""Small deterministic MCP server; default port 8001 avoids the Akasha API."""
from argparse import ArgumentParser
import os

def create_server(port=8001):
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("akasha-example", host="127.0.0.1", port=port,
                     streamable_http_path="/mcp")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def get_weather(city: str) -> str:
        """Return deterministic demo weather, not a live forecast."""
        return f"{city}: sunny (demo data)"
    return server

def main(argv=None):
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")))
    cli.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    args = cli.parse_args(argv)
    create_server(args.port).run(transport=args.transport)

if __name__ == "__main__":
    main()
