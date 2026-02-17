import asyncio
import os
import sys
from pathlib import Path

import akasha


async def _main() -> None:
    root = Path(__file__).resolve().parent
    server = root / "mcp_math_server.py"
    if not server.exists():
        raise FileNotFoundError(str(server))

    connection_info = {
        "math": {
            "command": sys.executable,
            "args": [os.fspath(server)],
            "transport": "stdio",
        }
    }

    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        temperature=0.0,
        verbose=True,
        keep_logs=False,
        env_file=os.fspath(root / ".env"),
    )

    # 注意：這裡會呼叫 LLM（需要你本機有對應 provider 的環境變數設定）。
    prompt = "2+3=?"
    print(agent.mcp_agent(connection_info, prompt))


if __name__ == "__main__":
    asyncio.run(_main())
