# Connect an agent to MCP tools

MCP (Model Context Protocol) lets an agent discover tools from an external MCP server. This tutorial uses a local deterministic server, so you can test the complete flow without a third-party account.

## 1. Install the dependencies

Install akasha and the MCP server package in the same virtual environment:

```bash
uv pip install "akasha-terminal[light]" mcp
```

Set a model provider key, for example:

```powershell
$env:OPENAI_API_KEY = "your_key"
```

## 2. Start the local MCP server

Create a file named `mcp_server.py` with the following complete server. It exposes two deterministic tools: `add` and `get_weather`.

```python
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
    """Return a deterministic weather response for a city."""
    return f"{city}: sunny"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Open Terminal 1 in the directory containing `mcp_server.py` and run:

```bash
python mcp_server.py
```

The server listens at:

```text
http://127.0.0.1:8000/mcp
```

Keep this terminal running.

## 3. Connect the Agent

Open Terminal 2 and create `mcp_agent.py`:

```python
import asyncio
import os

import akasha
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "example": {
                "transport": "streamable_http",
                "url": os.getenv(
                    "MCP_URL",
                    "http://127.0.0.1:8000/mcp",
                ),
            }
        },
        tool_name_prefix=True,
    )

    discovered_tools = await client.get_tools()
    tools = akasha.normalize_mcp_tools(discovered_tools)

    agent = akasha.agents(
        model=os.getenv("AKASHA_MCP_MODEL", "openai:gpt-4o-mini"),
        tools=tools,
        stream=False,
        max_round=4,
    )

    answer = await agent.acall(
        "Use the MCP add tool to add 20 and 22, then report the result."
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it in Terminal 2:

```bash
python mcp_agent.py
```

The expected answer should mention `42`. The exact wording depends on the selected model.

## 4. Understand the flow

```text
MCP server
    ↓ exposes tools
MultiServerMCPClient.get_tools()
    ↓ discovers tools
akasha.normalize_mcp_tools()
    ↓ normalizes tool results
akasha.agents(tools=tools)
    ↓ calls the selected MCP tool
Agent answer
```

## Local stdio and remote MCP

This example uses Streamable HTTP. akasha's MCP integration also supports local `stdio` servers. Use stdio when the Agent starts a local process; use Streamable HTTP when a separately managed server exposes an `/mcp` endpoint.

!!! warning
    An MCP server can expose powerful capabilities. Only connect to servers you trust, inspect the discovered tools, restrict network access where possible, and never pass secrets in prompts or tool arguments.

At this point the complete client and server code are both included in this page, so you can run the tutorial without opening another project file.
