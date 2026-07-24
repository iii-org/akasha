# Akasha 測試說明

本文件說明目前 `akasha-repo` 的測試環境、執行指令與測試案例。所有指令以 Windows `cmd.exe` 和專案外部的 `.venv` 為準；不要使用 WSL 或 PowerShell 執行這組測試。

目前測試範圍包含：

- Akasha 公開 API：`ask`、`agents`
- LLM provider adapter 與真實 provider contract
- stream / non-stream 行為
- thinking=False/True 與 semantic thinking level
- thinking、answer、tool event 的 response normalization
- logs 與 JSON serializable contract
- embedding provider contract
- RAG 的 Chroma、retrieval、grounded answer 與 cleanup pipeline

RAG/embedding live tests 是 opt-in，只有明確設定對應的 `RUN_*` 開關才會呼叫外部 provider。

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

Provider smoke 的 chat model 設定位於 `tests/smoke/test_provider_contract_smoke.py`；RAG embedding model 統一由 `tests/config/model_manifest.yaml` 的 `embeddings` 區段提供：

| Provider | Model alias | 必要設定 |
|---|---|---|
| OpenAI | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure:DeepSeek-V4-Flash` | `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_BASE_URL` |
| Gemini | `gemini:gemini-3.5-flash` | `GEMINI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama:gemma4:26b` | `OLLAMA_API_BASE` |

目前 manifest 中的 embedding cases：

| Provider | Embedding alias | 必要設定 |
|---|---|---|
| OpenAI | `openai:text-embedding-3-small` | `OPENAI_API_KEY` |
| Gemini | `gemini:gemini-embedding-2` | `GEMINI_API_KEY` |

若新增 embedding provider，請先更新 `model_manifest.yaml`；embedding contract test 會自動依 manifest 收集案例。

不要把 API key 寫入測試程式或 commit 到 Git。`tests/.env` 應維持在 ignored/private 狀態。

## 基本測試指令

### 1. 執行 unit tests

這是日常修改後的第一個回歸測試，不會呼叫真實 provider API：

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit -q"
```

重點包括：

- agent tools schema 與 agent 核心行為
- database structure 與文件處理
- crawler、encoding、prompt preprocessing
- retriever 與 search document normalization
- OpenAI/Azure/Gemini/Anthropic/Ollama adapter 設定
- thinking level normalization
- provider response 的 thinking/answer 欄位轉換
- ask/agents 公開 API event contract
- logging handler 行為

### 5. 執行 embedding provider smoke

這個測試會從 `tests/config/model_manifest.yaml` 讀取所有 embedding alias，實際呼叫 provider 並確認回傳向量數量、維度與數值格式：

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& set RUN_EMBEDDING_SMOKE=1&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_embedding_provider_contract.py -vv -s"
```

目前預期 2 個案例：OpenAI 與 Gemini。

### 6. 執行 OpenAI 分階段 RAG pipeline

這會實際執行：

```text
real embedding -> Chroma build -> Chroma reload -> retrieval -> chat model -> grounded answer -> cleanup
```

回答必須包含測試文件中的 `RAG-7319-TAIPEI`：

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& set RUN_RAG_PIPELINE=1&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_rag_pipeline_stages.py -vv -s"
```

### 7. 執行 Gemini 完整 RAG pipeline

這個測試會從 manifest 讀取 `gemini:gemini-embedding-2`，再實際串接 Gemini embedding、Chroma、retrieval 與 Gemini chat model：

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& set RUN_GEMINI_RAG=1&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_gemini_rag_pipeline.py -vv -s"
```

### 8. 執行 MCP smoke tests

MCP 測試使用本地 deterministic stdio server：

```text
tests/fixtures/mcp/echo_server.py
```

測試會驗證 MCP tool discovery、input schema、直接 tool invocation，以及由 `model_manifest.yaml` 的 `models` 區段逐一讀取每個實際模型進行 non-stream tool calling：

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& set RUN_MCP_SMOKE=1&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\smoke\test_mcp_pipeline.py -vv -s"
```

目前會執行 1 個 discovery case、每個 manifest model 1 個端到端 case，以及 1 個 stream guard case：

- discovery/direct invocation
- 每個 `models` 項目的 real agent non-stream tool calling
- async-only MCP agent 明確拒絕 sync stream，避免誤用

若 provider key 存在但無效，該 provider 的端到端 case 會失敗；這是刻意保留的真實 credential failure，不會被當成 skip。

### 2. 只執行 thinking adapter 測試

```bat
cmd.exe /d /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo && C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit\test_thinking_model_config.py -q"
```

### 3. 執行 API contract 與 response normalization 測試

```bat
cmd.exe /d /s /c "cd /d C:\Users\today\Projects\akasha-update\akasha-repo&& C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe -m pytest tests\unit\test_run_llm_chat_normalization.py tests\unit\test_thinking_model_config.py -q"
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

## RAG 測試資料與向量儲存

固定 RAG 測試資料位於：

```text
tests/tests_data/rag_smoke/
```

其中 `single_fact.txt` 包含固定識別碼 `RAG-7319-TAIPEI`，directory 測試資料包含 Alpha/Beta protocol facts。測試使用專案的 Chroma storage path，不使用正式資料庫或正式 memory 目錄。

RAG pipeline test 的 grounded assertion 分成兩層：

1. retriever 回傳的 `Document` 必須包含測試文件事實。
2. chat model 的最終回答也必須包含同一個固定識別碼。

因此只回傳非空文字、但沒有使用檢索內容的模型回答不會通過完整 pipeline test。

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

- MCP：請使用上方的 `RUN_MCP_SMOKE=1` 指令單獨執行。
- `tests/upgrade_tests/` 中的舊 RAG provider tests：仍需依各自 fixture 與 model 設定單獨執行。

不要用以下指令當作目前的非 RAG/MCP 回歸測試：

```bat
pytest tests
```

因為它可能收集尚未準備好的 legacy、RAG 或需要外部服務的測試。請優先使用本文件列出的明確路徑。

## 常見問題

### Live smoke tests 全部顯示 skipped

確認同一個 `cmd.exe` command 中有：

```bat
set RUN_PROVIDER_SMOKE=1
```

並確認使用的是正確的 `RUN_*` 開關；embedding/RAG 測試分別使用 `RUN_EMBEDDING_SMOKE`、`RUN_RAG_PIPELINE` 或 `RUN_GEMINI_RAG`。測試會對缺少 key 的 provider 個別 skip。

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

## Local total live test

若要在 local 端執行完整的 live API 測試，請先在 PowerShell 設定所有測試開關：

```powershell
$env:RUN_LLM_TESTS="1"
$env:RUN_PROVIDER_SMOKE="1"
$env:RUN_EMBEDDING_SMOKE="1"
$env:RUN_GEMINI_RAG="1"
$env:RUN_RAG_PIPELINE="1"
$env:RUN_RAG_SMOKE="1"
$env:RUN_MCP_SMOKE="1"
```

接著從 `akasha-repo` 目錄執行：

```powershell
& ..\.venv\Scripts\python.exe -m pytest tests -vv -s
```

這個 total command 的測試範圍是整個 `tests/`，包含：

- `tests/unit/`：不呼叫遠端模型的單元測試
- `tests/contract/`：公開 API、stream event、Agent response 與 logs contract
- `tests/smoke/`：OpenAI、Azure、Gemini、Anthropic、Ollama provider、embedding、RAG、web-info 與 MCP live smoke tests
- `tests/test_api_stability.py`：OpenAI/Gemini ask、RAG、Agent、vision、memory stability
- `tests/upgrade_tests/`：各 provider 的升級相容性測試
- 其他根目錄 `tests/test_*.py`：legacy API、DB、evaluation、Agent 與 summary 測試

各開關的作用如下：

這些開關只控制標示為 live/integration 的那一小組測試，不是整個
`tests/` 的總開關。大多數測試以不區分大小寫的 `1`、`true` 或 `yes` 啟用；
`RUN_LLM_TESTS` 所涵蓋的 `test_agents_final_action_integration.py` 目前只判斷
變數是否為非空，因此請統一使用 `=1`，不要依賴其他值。

### Live 測試矩陣

| 開關 | 實際測試檔與案例 | Chat model | Embedding / 外部元件 | 驗證內容 |
|---|---|---|---|---|
| `RUN_LLM_TESTS=1` | `test_live_gemini_agent.py`：3 cases | `gemini:gemini-2.5-flash` | 無工具；Gemini live API | Agent final answer、stream 的 `thinking`/`answer` event，以及 `thinking=False` 時不誤用 budget |
| 〃 | `test_agents_final_action_integration.py`：每個模型 1 case | `openai:gpt-4o`、`gemini:gemini-2.5-flash` | 無工具 | 真實模型回傳的 final-action alias 能被 Agent 接受 |
| 〃 | `smoke/test_web_info_ask.py`：1 case | `gemini:gemini-3.5-flash` | web-info 來源 | `ask` 能根據 web-info 回答 Akasha 相關問題 |
| 〃 | `smoke/test_ask_info_urls_pure.py`：1 case | `gemini:gemini-2.5-flash` | 兩個 URL `info` 來源 | 公開 `ask` API 能接受多個 URL reference |
| `RUN_PROVIDER_SMOKE=1` | `smoke/test_provider_contract_smoke.py`：目前 5 models × 6 基本/ thinking cases，加上 Gemini 2 個 native thinking cases，共 32 cases | 由 manifest 的 5 個 model：OpenAI `gpt-5.4`、Azure `DeepSeek-V4-Flash`、Gemini `gemini-3.5-flash`、Anthropic `claude-sonnet-4-6`、Ollama `gemma4:26b` | 各 provider 的真實 chat endpoint | `ask`/`agents` 的 non-stream、stream、logs JSON serializable，以及 thinking/answer 分離與 native thinking 設定 |
| `RUN_EMBEDDING_SMOKE=1` | `smoke/test_embedding_provider_contract.py`：manifest 的每個 embedding 1 case，目前 2 cases | 無 chat model | OpenAI `text-embedding-3-small`、Gemini `gemini-embedding-2` | 向量數量、維度、數值格式與 provider embedding contract |
| `RUN_RAG_PIPELINE=1` | `smoke/test_rag_pipeline_stages.py`：4 stages | `openai:gpt-5.4` | OpenAI `text-embedding-3-small` + Chroma | Chroma reload → retrieval → chat model grounded answer → cleanup；答案須含 `RAG-7319-TAIPEI` |
| `RUN_GEMINI_RAG=1` | `smoke/test_gemini_rag_pipeline.py`：1 end-to-end case | `gemini:gemini-2.5-flash` | manifest 的 Gemini `gemini-embedding-2` + Chroma | Gemini embedding、Chroma 建立/搜尋與 grounded chat answer |
| `RUN_RAG_SMOKE=1` | `smoke/test_rag_contract_smoke.py`：OpenAI file、Gemini directory、Windows absolute path，共 3 cases（非 Windows 的 path case 會再 skip） | OpenAI `gpt-5.4`、Gemini `gemini-3.5-flash` | 對應 OpenAI/Gemini embedding + RAG file/path input | `RAG` 公開介面、文件載入、retrieval、非空回答與 logs JSON contract |
| `RUN_MCP_SMOKE=1` | `smoke/test_mcp_pipeline.py`：1 discovery、每個 manifest model 1 agent case、1 stream guard；目前共 7 cases | manifest 的 5 個 chat models | 本地 `tests/fixtures/mcp/echo_server.py` | MCP tool discovery/schema/direct invocation、每個真實模型的 non-stream Agent tool calling，以及明確拒絕不支援的 sync stream |

其中 `RUN_LLM_TESTS` 不會啟用 `RUN_PROVIDER_SMOKE`、任何 RAG、embedding 或 MCP
測試；反過來設定 `RUN_PROVIDER_SMOKE` 也不會啟用一般的 Gemini Agent/web-info
測試。要執行哪一格，必須設定該格自己的開關。

### 開關未設定時會發生什麼

- 對應的 live 測試會在 pytest collection 時以 `SKIPPED` 結束，不會呼叫模型、抓 web-info、建立向量庫或啟動 MCP agent；這不是測試通過（passed），而是明確表示該 live contract 沒有執行。
- 開關已設定但該 provider 的必要 key 不存在時，通常只 skip 該 provider 的案例。例如 provider smoke 可以只執行已設定 key 的模型；embedding/RAG/MCP 也會依 provider 個別 skip。
- 開關已設定且 key 存在但無效、endpoint 不可連線或模型名稱錯誤時，案例會 fail；這是刻意保留的 credential/endpoint failure，不會被當成 skip。
- 沒有設定任何開關時，`tests/unit` 與 `tests/contract` 仍照常執行；它們使用 mock/fake model，不需要 live API。這些開關也不會替根目錄的 legacy `tests/test_*.py` 或 `tests/upgrade_tests/` 自動加上保護，因此不要把 `pytest tests` 解讀成「所有外部 API 都已由開關控制」。

因此，若只執行日常本地回歸，建議使用 `tests/unit`、`tests/contract`；若要驗證真實 provider，除了設定對應開關，也要確認 manifest 的 model ID 與必要 secrets 都可用。

API keys 預設從 `tests/.env` 載入；也可以透過 `ENV_FILE` 指定其他 `.env`：

```powershell
$env:ENV_FILE="C:\path\to\tests\.env"
```

注意：這個 total command 會實際消耗各家模型 API quota。`MCP` 在 Windows 可能受 named pipe 權限影響，Ollama 則需要可連線的 `OLLAMA_API_BASE`；這兩類失敗不一定代表 OpenAI/Gemini/Anthropic provider 本身有問題。