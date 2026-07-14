# Akasha 測試說明

本文件說明目前 `akasha-repo` 的測試環境、執行指令與測試案例。所有指令以 Windows `cmd.exe` 和專案外部的 `.venv` 為準；不要使用 WSL 或 PowerShell 執行這組測試。

目前測試範圍先聚焦在：

- Akasha 公開 API：`ask`、`agents`
- LLM provider adapter 與真實 provider contract
- stream / non-stream 行為
- thinking=False/True 與 semantic thinking level
- thinking、answer、tool event 的 response normalization
- logs 與 JSON serializable contract

RAG、embedding 與 MCP 測試目前刻意不納入 provider smoke workflow，等測試資料準備完成後再加入。

## 測試環境

工作目錄：

```text
C:\Users\today\Projects\akasha-update\akasha-repo
```

Python 執行檔：

```text
C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe
```

所有測試指令都可以使用以下形式：

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest <test-path>"
```

## 環境變數與 secrets

真實 provider smoke test 使用：

```text
tests\.env
```

目前 provider model 設定位於 `tests/smoke/test_provider_contract_smoke.py`：

| Provider | Model alias | 必要設定 |
|---|---|---|
| OpenAI | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure:DeepSeek-V4-Flash` | `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_BASE_URL` |
| Gemini | `gemini:gemini-3.5-flash` | `GEMINI_API_KEY` |
| Anthropic | `anthropic:claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama:gemma4:26b` | `OLLAMA_API_BASE` |

不要把 API key 寫入測試程式或 commit 到 Git。`tests/.env` 應維持在 ignored/private 狀態。

## 基本測試指令

### 1. 執行所有 unit 與 contract tests

這是日常修改後的第一個回歸測試，不會呼叫真實 provider API：

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit tests\contract -q"
```

涵蓋目前約 58 個測試案例，重點包括：

- agent tools schema 與 agent 核心行為
- database structure 與文件處理
- crawler、encoding、prompt preprocessing
- retriever 與 search document normalization
- OpenAI/Azure/Gemini/Anthropic/Ollama adapter 設定
- thinking level normalization
- provider response 的 thinking/answer 欄位轉換
- ask/agents 公開 API event contract
- logging handler 行為

### 2. 只執行 thinking adapter 測試

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit\test_thinking_model_config.py -q"
```

### 3. 執行 API contract 與 response normalization 測試

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\contract tests\unit\test_run_llm_chat_normalization.py -q"
```

### 4. 執行 logging 回歸測試

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit\test_logging_config_unit.py -q"
```

## 真實 provider smoke tests

這組測試會真的呼叫 provider，可能產生 API 費用或消耗 Ollama server 資源。只有明確設定 `RUN_PROVIDER_SMOKE=1` 才會執行。

### 執行全部 provider smoke tests

```bat
cmd.exe /d /c "set RUN_PROVIDER_SMOKE=1 && cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_provider_contract_smoke.py -q"
```

如果使用 `set` 搭配 `&`，請使用引號包住變數設定，避免 cmd.exe 把尾端空白放入變數：

```bat
set "RUN_PROVIDER_SMOKE=1"
```

### 只測某個 provider

```bat
cmd.exe /d /c "set RUN_PROVIDER_SMOKE=1 && cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_provider_contract_smoke.py -q -k ollama"
```

可將 `ollama` 替換成 `openai`、`azure`、`gemini` 或 `anthropic`。

### Provider smoke 測試案例

每個 provider 基本上會測試以下 4 種 contract：

1. `ask(stream=False, thinking=False)`：確認非串流 ask 能收到非空文字答案。
2. `ask(stream=True, thinking=False)`：確認串流 ask 只產生文字 chunks。
3. `agents(stream=False, thinking=False)`：確認 agent 能收到 final answer，且 logs 可 JSON serialize。
4. `agents(stream=True, thinking=False)`：確認 agent stream event 能正規化為 `answer`、`tool` 或 `warning`。

另外有 thinking=True 測試：

5. `ask(stream=True, thinking=True, thinking_budget="medium")`：確認 provider 的 thinking 與 answer 可被 Akasha 分離並正常回傳。
6. `ask(stream=True, thinking=True, thinking_budget="medium")`：確認 thinking-enabled stream 仍能產生 final answer。

Gemini 另有 native thinking stream contract 測試。完整收集時目前共 32 個 provider smoke cases；未設定必要 API key 的 provider 會 skip。

## Thinking semantic level

前端統一使用：

```python
thinking=True
thinking_budget="low"       # 或 "medium"、"high"
```

目前換算規則依 `max_output_tokens` 動態計算：

```text
low    = max(2048, max_output_tokens * 0.5)
medium = max(4096, max_output_tokens * 1)
high   = max(8192, max_output_tokens * 2)
```

例如：

```python
max_output_tokens=65536
thinking_budget="high"
```

會得到 131072 的 numeric thinking budget。整數仍可作為進階覆寫：

```python
thinking_budget=32768
```

各 adapter 的實際轉換不同：

- OpenAI/Azure：`low`、`medium`、`high` 轉為 `reasoning_effort`。
- Gemini：轉為 numeric `thinking_budget`。
- Anthropic：轉為 `budget_tokens`。
- Ollama：`num_predict` 會包含 thinking budget 與 answer `max_output_tokens`，並維持至少 2048。

## 測試輸出與 logs

使用 `verbose=True` 時，ask/agents 會顯示：

```text
Thinking: True
Thinking budget level: medium
Effective thinking budget: 4096
```

使用 `keep_logs=True` 時，每次呼叫的 log metadata 會包含：

```python
{
    "thinking": True,
    "thinking_budget_level": "medium",
    "effective_thinking_budget": 4096,
}
```

若 provider 回傳 thinking，實際 thinking 內容會放在 logs 的 `thinking` 欄位；stream API 則會產生 `thinking` 與 `answer` event。

## 目前暫不執行的測試

以下測試不屬於目前的 provider contract smoke workflow：

- RAG：需要文件、embedding、vector store 與檢索資料。
- MCP：需要 MCP server、工具 schema 與測試資料。
- `tests/upgrade_tests/` 中的 RAG provider tests：等 RAG 測試資料準備完成後再執行。

不要用以下指令當作目前的非 RAG/MCP 回歸測試：

```bat
pytest tests
```

因為它可能收集尚未準備好的 legacy、RAG 或需要外部服務的測試。請優先使用本文件列出的明確路徑。

## 常見問題

### Smoke tests 全部顯示 skipped

確認同一個 `cmd.exe` command 中有：

```bat
set RUN_PROVIDER_SMOKE=1
```

並確認 `tests/.env` 存在必要 key。測試會對缺少 key 的 provider 個別 skip。

### Ollama 連線失敗

先確認 `OLLAMA_API_BASE` 是 tunnel 的根 URL，例如：

```text
https://ollama.example.org
```

不要自行加上 `:11434`，也不要把 `/api/chat` 寫入 base URL。可先確認：

```bat
curl https://ollama.example.org/
curl https://ollama.example.org/api/version
curl https://ollama.example.org/api/tags
```

並確認遠端模型名稱完全一致：

```text
gemma4:26b
```

### API key 或 endpoint 設定錯誤

- public OpenAI 使用 `OPENAI_API_KEY`。
- Azure OpenAI 使用 `AZURE_OPENAI_API_KEY` 與 `AZURE_OPENAI_BASE_URL`。
- 不要用 Azure endpoint 當作 `OPENAI_BASE_URL`。
- 不要把完整 `/chat/completions` 路徑重複傳給 OpenAI-compatible SDK。
