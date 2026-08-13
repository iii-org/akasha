"""Connect an Akasha agent to a Streamable HTTP MCP server.

Start the server in another terminal first::

    python examples/mcp_server.py

Set the model provider key required by your selected model, then run::

    python examples/ex_mcp.py
"""

from __future__ import annotations

import asyncio
import os

import akasha
from langchain_mcp_adapters.client import MultiServerMCPClient


async def run_agent() -> str:
    client = MultiServerMCPClient(
        {
            "example": {
                "transport": "streamable_http",
                "url": os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp"),
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
        thinking=False,
        keep_logs=True,
        max_round=4,
    )
    response = await agent.acall(
        "Use the MCP add tool to add 20 and 22, then report the result."
    )
    agent.save_logs("logs_mcp_agent.json")
    return response


def main() -> None:
    print(asyncio.run(run_agent()))


if __name__ == "__main__":
    main()
