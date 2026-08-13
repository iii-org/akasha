# 將 Agent 連接到 MCP 工具

MCP（Model Context Protocol）可以讓 Agent 從外部 MCP Server 探索工具。本篇使用本機的固定結果 Server，因此不需要第三方帳號就能測試完整流程。

## 1. 安裝依賴

在同一個虛擬環境安裝 akasha 與 MCP Server 套件：

```bash
uv pip install "akasha-terminal[light]" mcp
```

設定模型 Provider 的 key，例如：

```powershell
$env:OPENAI_API_KEY = "your_key"
```

## 2. 啟動本機 MCP Server

Repository 已提供 `examples/mcp_server.py`，其中有兩個固定結果工具：`add` 與 `get_weather`。

在 Terminal 1 切換到 repository 根目錄後執行：

```bash
python examples/mcp_server.py
```

Server 會監聽：

```text
http://127.0.0.1:8000/mcp
```

請保持這個終端機執行中。

## 3. 連接 Agent

開啟 Terminal 2，建立 `mcp_agent.py`：

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
        "請使用 MCP add 工具計算 20 加 22，然後回報結果。"
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

在 Terminal 2 執行：

```bash
python mcp_agent.py
```

預期回答應該會提到 `42`，但實際文字會依選用模型而不同。

## 4. 理解執行流程

```text
MCP Server
    ↓ 提供工具
MultiServerMCPClient.get_tools()
    ↓ 探索工具
akasha.normalize_mcp_tools()
    ↓ 正規化工具結果
akasha.agents(tools=tools)
    ↓ 呼叫選定的 MCP 工具
Agent 回答
```

## 本機 stdio 與遠端 MCP

本篇使用 Streamable HTTP。akasha 的 MCP 整合也支援本機 `stdio` Server：當 Agent 需要自行啟動本機程序時使用 stdio；當獨立管理的 Server 提供 `/mcp` endpoint 時，使用 Streamable HTTP。

!!! warning
    MCP Server 可能提供強大的操作能力。只連接信任的 Server，先檢查探索到的工具，盡可能限制網路存取，也不要把秘密放在 Prompt 或工具參數中。

Repository 完整範例是 [`examples/ex_mcp.py`](https://github.com/iii-org/akasha/blob/master/examples/ex_mcp.py)，本機 Server 是 [`examples/mcp_server.py`](https://github.com/iii-org/akasha/blob/master/examples/mcp_server.py)。
