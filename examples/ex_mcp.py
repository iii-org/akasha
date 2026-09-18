"""Connect to examples/mcp_server.py using Streamable HTTP and async tools."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

import asyncio
import os

async def run_agent(args):
    import akasha
    from langchain_mcp_adapters.client import MultiServerMCPClient
    url = args.url or os.getenv("MCP_URL", "http://127.0.0.1:8001/mcp")
    client = MultiServerMCPClient(
        {"example": {"transport": "streamable_http", "url": url}}, tool_name_prefix=True)
    tools = akasha.normalize_mcp_tools(await client.get_tools())
    agent = akasha.agents(model=args.model, tools=tools, env_file=args.env_file,
                          stream=False, verbose=True, keep_logs=True, max_round=4)
    # MCP tools are async-only. Do not use the synchronous stream path.
    response = await agent.acall("Use the MCP add tool to add 20 and 22, then report the result.")
    agent.save_logs("mcp-agent.json")
    return response

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--url", help="MCP endpoint; defaults to MCP_URL or http://127.0.0.1:8001/mcp")
    args = configure(cli, argv)
    # Retain the former example-specific model override.
    if not cli.parse_args(argv).model and os.getenv("AKASHA_MCP_MODEL"):
        args.model = os.environ["AKASHA_MCP_MODEL"]
    with workspace(args, "mcp"):
        asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
