# akasha-repo 測試補強計畫

Date: 2026-07-14

## 目的

這份文件說明如何把現有測試從「可跑」補強成「能保護改動」：

- 降低 API / 模型升級後才發現問題的風險
- 減少 mock 過多造成的假安全感
- 對 `akasha.ask()`、`akasha.agents()`、`akasha.RAG()` 這類 public API 建立契約保護
- 讓 Windows path、stream、thinking、logs、env 這些高風險路徑有回歸保障

## 核心原則

### 1. mock 只留給純邏輯

適合 mock 的範圍：

- 參數正規化
- 路徑 / token / prompt 轉換
- 錯誤處理
- helper function

不適合只靠 mock 的範圍：

- provider adapter
- chat model API shape
- streaming event 格式
- thinking / tool calling 行為
- logs / env 讀取 / 回傳型別

### 2. public API 要有真實 smoke

至少保留少量真實 provider 測試，專門驗證：

- `ask()`
- `agents()`
- `RAG()`
- `summary()`
- `MemoryManager`

這些測試不追求完整覆蓋，而是確認「主要使用路徑仍可運作」。

### 3. 測「關鍵組合」，不要全排列

參數很多時，不要測所有組合；優先測：

- 預設值
- 常用值
- 最容易壞的值
- 不同 provider 的最小 smoke

例如 `ask()` 可優先覆蓋：

- `stream=False`, `thinking=False`
- `stream=True`, `thinking=True`
- `keep_logs=True`
- `max_output_tokens`
- `env_file`

## 測試分層建議

### A. Unit tests

位置：`tests/unit/`

目標：

- 不依賴 API key
- 不碰真模型
- 只驗證內部邏輯與錯誤處理

適合放的案例：

- `db_structure`
- `search_doc`
- `run_llm` normalization
- prompt handling
- path sanitization
- alias / mapping / helper logic

### B. Contract / smoke tests

位置建議：`tests/test_*.py` 或新建 `tests/smoke/`

目標：

- 真實呼叫 provider
- 驗證 public API 行為
- 驗證回傳 shape、logs、thinking、streaming、tool calling

適合放的案例：

- `akasha.ask()` 最小問答
- `akasha.agents()` tool calling + stream
- `akasha.RAG()` 文件檢索
- `MemoryManager` 寫入與查詢
- `summary()` 的文件摘要

### C. Upgrade tests

位置：`tests/upgrade_tests/`

目標：

- 升級依賴後快速發現 provider breakage
- 保留少量但高價值的真實 smoke

## 建議的測試矩陣

### 依功能切分

#### ask

最少覆蓋：

- `stream=False`
- `stream=True`
- `thinking=True`
- `thinking=False`
- `keep_logs=True`
- `max_output_tokens`
- `env_file`

#### agents

最少覆蓋：

- tool calling
- stream events
- thinking events
- final answer 為 `str`
- logs 可序列化

#### RAG

最少覆蓋：

- file input
- directory input
- Windows absolute path
- `openai:` / `gemini:` 至少各一條 smoke

#### MemoryManager

最少覆蓋：

- add / search
- persistence path
- Windows absolute path

#### summary

最少覆蓋：

- text input
- file input
- URL input
- 回傳型別與基本內容

## 真實 smoke 的建議規模

建議保留以下最小集合：

- OpenAI：1 條
- Gemini：1 條
- Anthropic：1 條
- Ollama：若支援，再 1 條

每個 provider 不需要測很多功能，但至少要跑一次最核心路徑。

## 建議的落地順序

### P0

1. 補 `ask()` 參數組合 smoke
2. 補 `agents()` streaming / thinking / tool calling
3. 補 `RAG()` 的 provider / path smoke
4. 補 `MemoryManager` 的持久化與查詢
5. 補 Windows absolute path regression

### P1

1. 補 `summary()` smoke
2. 補 provider alias 與 env 來源測試
3. 補 logs / event schema contract tests

### P2

1. 整理更多純邏輯 helper unit test
2. 清理過度依賴 mock 的舊測試
3. 將高度重複的 integration 測試收斂成共用 fixture

## 建議實作方式

### 1. 先保留 unit，補 smoke

不要先大改舊測試；先在現有測試旁新增少量 smoke，讓 CI 有真實保護。

### 2. 再收斂 mock

把只是在驗證「函數有被呼叫」的測試，逐步改成：

- 驗證輸入輸出
- 驗證 event shape
- 驗證例外

### 3. 統一測試入口

建議讓以下測試維持穩定命名與 marker：

- `unit`
- `integration`
- `upgrade`
- `requires_api`
- `smoke`

## 可直接照做的檢查清單

- [ ] `ask()` 至少有 2 條真實 smoke
- [ ] `agents()` 至少有 1 條 stream + thinking smoke
- [ ] `RAG()` 至少有 1 條 file / dir smoke
- [ ] `MemoryManager` 有 persistence smoke
- [ ] Windows absolute path 有 regression test
- [ ] `summary()` 有最小真實 smoke
- [ ] provider alias 與 env file 行為有測試
- [ ] 重要 bug 都有對應 regression test

## 驗收標準

做到以下狀態，就代表補強有成效：

- 真模型或真 provider 的 smoke 能在升級時提早抓到 breakage
- public API 參數新增或改名時，不會只靠人工發現
- Windows path、stream、thinking、logs、env 這些高風險點有固定保護
- mock 測試不再承擔整體可用性的責任

## 本次決議與後續建置規則

開始實作前，先依照 [2026-07-testing-preparation-checklist.md](2026-07-testing-preparation-checklist.md) 逐項完成 API、模型、資料、RAG、CI 與診斷準備。清單全部完成後，再開始修改 production code 與建立完整測試案例。

本次測試目標是驗證模型與 provider 能正確接線，不評估模型回答品質；但測試仍必須完成可驗證的行為，例如工具確實被選取與執行、RAG 確實完成檢索流程、response/event 符合公共契約。

### Capability manifest 不等於 skip 清單

每個 model manifest 項目都要定義 capability 的 `native`、`behavior` 與 `fallback`。宣告不支援的功能仍要故意傳入測試，驗證程式是否正確 warning、降級或回傳 unsupported notice。

- `native: true`：功能必須成功，否則測試失敗。
- `native: false`：必須得到預期的 `unsupported-and-downgraded` 或 `unsupported-notice`，不能直接 skip。
- manifest 與實際 provider 行為不一致時，必須明確報告，不得靜默修正或忽略。

測試結果至少區分：

- `supported-and-passed`
- `unsupported-and-downgraded`
- `declared-supported-but-failed`
- `unsupported-but-not-handled`
- `response-contract-violated`

### 統一 stream event contract

- `stream=False` 回傳 `str`。
- `stream=True` 回傳 event iterator。
- event 至少包含 `answer`、`thinking`、`tool`、`warning`，且可 JSON 序列化。
- `thinking=False` 也使用 `{"type": "answer", "data": "..."}`，不能因 thinking 開關改變 iterator 元素型態。
- provider 不支援原生 stream 時，可以先取得完整答案，但仍要透過 event iterator 回傳，避免 fallback 破壞 public API 型態。
- 舊版 `Iterator[str]` 若需保留，使用明確 legacy mode 與 deprecation policy，不讓型態隨 provider 隱式改變。

### RAG 準備與驗收

RAG 測試必須使用小型固定文件、embedding manifest、獨立 temporary Chroma/vector store、UTF-8/中文 fixture，以及 Windows absolute path 的 runner 或等價 regression strategy。RAG 不做所有 chat model × embedding model 全排列，而由 manifest 指定最小同 provider 與跨 provider 組合。

RAG 驗收重點是流程成功，不是回答品質；至少分別確認 document load、chunk、embedding、vector store write、retrieval、chat invoke 與 response normalization。測試應能觀察 retrieved documents 或等價 diagnostics，並驗證 vector 維度、metadata 與 logs 可序列化。

### Live smoke 成本與診斷

PR 可以呼叫真實 API，但使用短 prompt、`temperature=0`、有限 `max_output_tokens`、timeout 與有限 retry。retry 只處理可辨識的 provider transient error，不能把 assertion error 當作 API retry。

每次 live test 失敗時，保存 provider、model、embedding model、Python 與套件版本、requested/effective configuration、request mode、capability status、fallback 與 response/event schema 摘要；不得保存 API key、Authorization header 或不必要的敏感 prompt。

完整的事前準備、fixture、tool-calling、RAG、CI 與回報格式，統一收錄在上述 checklist，之後新 session 應以該文件作為施工入口。

### Embedding model 是獨立的測試對象

Embedding model 必須有自己的 manifest，不可只作為 RAG 的一個字串參數。manifest 至少記錄：

- provider 與完整 embedding model ID
- API key / environment source
- 維度或維度驗證方式
- PR、nightly、release 分組
- 可搭配的 chat model 組合

Azure OpenAI deployment 在 manifest 中使用獨立的 azure:<deployment-name> alias；一般 OpenAI 使用 openai:<model-id>。兩者都走 OpenAI-compatible ChatModel factory，但 API key、base URL、provider label 與測試報告必須分開。

Embedding contract tests 要確認 alias 可建立、文字可轉成非空 vector、vector 維度一致、query/document 維度相容、可寫入 vector store、可執行 similarity search，以及錯誤能被明確分類。RAG 與 MemoryManager 都必須至少各有 embedding smoke；RAG 還要覆蓋同 provider 與跨 provider 的最小組合，不做無限制全排列。

### MCP 是 agent tool integration 的獨立路徑

目前 MCP 使用方式是由 langchain-mcp-adapters 連接 MCP server、取得 MCP tools，再傳入 akasha.agents(tools=...)。因此 MCP 測試不能只測本地 BaseTool；至少要驗證：

- MCP server tool discovery 成功
- MCP tool name、description、input schema 可被 agent 接收
- 模型選擇正確的 MCP tool
- MCP tool 收到正確參數並實際執行
- tool result 能回到 agent
- non-stream 與 stream 的 tool / answer event 符合契約
- MCP tool call 與結果可寫入 logs 並 JSON serialize
- MCP server unavailable、tool error、錯誤 schema 有獨立診斷訊息

第一階段使用本地 deterministic MCP server 與 stdio transport，不需要另外申請 MCP API key。Streamable HTTP 使用本地 `/mcp` endpoint 加入 deterministic contract test；舊 HTTP+SSE 不作為新的測試主路徑，也不要把外部 MCP 服務的不穩定性混入 PR 基本 smoke。

