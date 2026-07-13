# LangChain 1.3+ 原生 Agent 升級說明

## Summary

Akasha 將 Agent 從自製 JSON ReAct loop 升級為 LangChain 1.3+ `create_agent` 與原生 tool calling。`akasha.agents(...)` 的非串流公開回傳維持 `str`；串流模式則提供共通事件格式。內部使用 LangChain message/state，統一處理 answer、thinking、tool calls 與 logs。

## Design decisions

- 支援 OpenAI、Gemini、Ollama 與 Anthropic ChatModel。
- 保留 `openai:<model>`、`gemini:<model>`、`ollama:<model>`、`anthropic:<model>` 模型字串格式。
- `handle_model()` 同時接受模型字串與 LangChain ChatModel 物件。
- Agent 正常路徑使用 `create_agent`，不保留自製 JSON ReAct fallback。
- 非串流 `akasha.agents(...)` 仍回傳 final answer 字串。
- `ask()` 與 `agents()` 都接受 `thinking` 與 `thinking_budget`。
- `thinking=True` 啟用 provider-specific thinking；搭配 `stream=True` 時自動公開 thinking event，不需要開發者另外傳入 `include_thinking=True`。
- `thinking=False` 時 `thinking_budget` 自動失效並被忽略，不會因為同時提供 budget 而初始化失敗。
- `thinking_budget` 只限制思考 token；`max_output_tokens` 只限制最終回答 token，兩者是獨立額度。
- provider reasoning/thinking 不混入 answer；非串流時寫入 logs。
- `include_thinking` 仍可作為 Agent 串流顯示覆寫，但一般使用不需要再設定。

## Implementation

- 使用 `ChatOpenAI`、`ChatGoogleGenerativeAI`、`ChatOllama`、`ChatAnthropic`。
- 以 LangChain `AIMessage`、`AIMessageChunk`、`ToolMessage` 與 agent state 作為共通資料來源。
- 補齊 `BaseTool` 的參數 schema，確保 native tool calling 能取得可靠輸入。
- 移除 Agent 的 JSON action parser、手動 Thought/Action/Observation loop 與手動 tool retry loop。
- 從 agent state 的最後 AI message 取出 final answer；串流 Agent 使用 LangGraph `stream_mode="messages"` 取得 `AIMessageChunk`，提供 token/chunk-level events。
- logs 保存 answer、thinking、provider、model、tool calls、messages 與可序列化 metadata。
- Gemini model string 會將 `thinking=True` 轉成 `include_thoughts=True`，並將 `thinking_budget` 轉成 Gemini 的同名設定。
- Gemini model string 支援 `thinking=True` 與 `thinking_budget`。目前 OpenAI、Anthropic、Ollama 的字串模型尚未建立統一的 provider-specific thinking mapping；若以字串啟用 thinking，會明確回報錯誤。直接傳入的 LangChain ChatModel 必須由開發者自行完成 thinking 設定。

## Streaming contract

`agents(..., stream=True)` 與 `ask(..., stream=True, thinking=True)` 使用固定事件格式。Agent 底層使用 LangGraph `stream_mode="messages"`，因此 answer/thinking 是逐 chunk 產生；實際網路傳輸仍可能將多個 chunk 批次送回。

```python
{"type": "answer", "data": "..."}
{"type": "tool", "data": {...}}
{"type": "thinking", "data": "..."}  # thinking=True 且 stream=True
```

最小用法：

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=True,
    thinking=True,
    thinking_budget=1024,
    max_output_tokens=1024,
)

for event in agent("請比較向量資料庫與關聯式資料庫"):
    if event["type"] == "thinking":
        print(event["data"], end="", flush=True)
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

`ask()` 在 `thinking=True` 的串流模式也回傳相同事件格式；`thinking=False` 則維持既有字串 chunk 介面，以保持 API 相容性。非串流 `ask()` 與 `agents()` 均回傳 `str`。

## Thinking provider mapping

Gemini 的 model string 會建立等價設定：

```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    thinking_budget=1024,
    include_thoughts=True,
)
```

`thinking` 內容只進入 thinking events 與 logs，不會併入公開 answer。若 provider 沒有可用的 mapping，Akasha 應明確報錯，不應假裝已啟用 thinking。

## Testing and acceptance

- 四種 provider 的 ChatModel factory 與 native tool calling 路徑有單元測試；真實 API smoke test 目前以 Gemini 為主。
- Agent final answer 維持非空 `str`。
- 驗證工具 schema、工具錯誤、多輪工具、多工具及平行工具呼叫。
- 驗證 thinking 不污染 answer，且 LangChain message/state 能正確寫入 logs。
- 驗證非串流與結構化串流行為，以及不支援 tool calling 時的明確錯誤。
- 依賴與測試以 LangChain `>=1.3,<2.0` 為基準。

目前 Windows `.venv` 驗證結果：

- unit tests：`41 passed`
- Gemini live integration tests：`3 passed`
- live tests 需要 `RUN_LLM_TESTS=1`，會實際呼叫 API 並產生費用；測試案例位於 `tests/test_live_gemini_agent.py`。
