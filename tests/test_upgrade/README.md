這個資料夾用來驗證升級 `langchain-mcp-adapters (>=0.2)` 後，Akasha 的 MCP client 整合仍可正常運作。

## 內容
- `mcp_math_server.py`: 一個最小可用的 MCP server（stdio transport），提供 `add`、`multiply` 工具。
- `mcp_client_example.py`: 用 `akasha.agents().mcp_agent(...)` 連線並取得工具的簡單示例（LLM 使用 `gemini:gemini-2.5-flash`，API token 讀取 `tests/test_upgrade/.env`）。

## 手動跑範例（可選）
1) 啟動 venv

   `source /Users/today/Projects/Envs/akasha-upgrade/.venv/bin/activate`

2) 跑 client 範例（它會自動用 subprocess 起 `mcp_math_server.py`）

   `python tests/test_upgrade/mcp_client_example.py`
