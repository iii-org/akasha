# public API 測試矩陣對照表

Date: 2026-07-14

## 目的

這份文件把前一版的 public API 測試矩陣，直接對應到現有 `tests/` 裡的檔案，方便你快速看出：

- 哪些 API 已經有測試
- 哪些測試偏 unit / smoke / regression
- 哪些矩陣項目仍有缺口

## 對照總表

| API / 能力 | 現有測試檔案 | 目前覆蓋重點 | 缺口 / 備註 |
|---|---|---|---|
| `akasha.ask()` | `tests/test_akasha.py`、`tests/test_live_gemini_agent.py` | 基本問答、stream、thinking、logs、live Gemini | 缺少多 provider 矩陣；`keep_logs` / `env_file` / `max_output_tokens` 沒有完整排列 |
| `akasha.agents()` | `tests/test_agent.py`、`tests/test_live_gemini_agent.py`、`tests/unit/test_agents_core.py`、`tests/test_agents_observation_normalization.py`、`tests/test_agents_final_action_integration.py` | tool calling、final answer、message normalization、stream/thinking（live） | 缺少跨 provider smoke；多工具與 logs 契約可再補強 |
| `akasha.RAG()` | `tests/test_akasha.py`、`tests/test_api_stability.py`、`tests/upgrade_tests/test_openai_rag.py`、`tests/upgrade_tests/test_gemini_rag.py`、`tests/upgrade_tests/test_light_restrictions.py` | OpenAI / Gemini RAG smoke、Windows absolute path、light restriction | 缺 Anthropic/Ollama smoke；`search_type` 矩陣覆蓋有限 |
| `akasha.summary()` | `tests/test_summary.py`、`tests/unit/test_preprocess_prompts.py`、`tests/test_api_stability.py` | map_reduce 摘要、基本回傳型別、部分 prompt 處理 | 缺 text/file/URL 的完整矩陣與 refine smoke |
| `MemoryManager` | `tests/test_akasha.py`、`tests/test_api_stability.py` | memory 寫入、查詢、Windows absolute path | 缺 persistence 重啟 smoke 與 provider alias 矩陣 |
| provider alias / factory | `tests/unit/test_handle_objects_unit.py`、`tests/unit/test_run_llm_chat_normalization.py`、`tests/unit/test_thinking_model_config.py`、`tests/unit/test_agents_core.py` | 參數正規化、thinking 設定、chat input 轉換 | 缺少真實 provider 契約測試，主要還是 unit |
| logs / keep_logs | `tests/test_akasha.py`、`tests/test_agent.py`、`tests/test_summary.py`、`tests/unit/test_logging_config_unit.py`、`tests/test_logging_behavior.py` | logging 基礎、keep_logs 行為、console/file handler | 缺 public API 層的完整 logs schema 驗證 |
| Windows absolute path | `tests/test_api_stability.py`、`tests/unit/test_db_structure.py` | `RAG(data_source=<absolute path>)`、`MemoryManager(memory_dirname=<absolute path>)` | 已有 regression，建議保持為 P0 |
| light / full 安裝邊界 | `tests/upgrade_tests/test_light_restrictions.py`、`tests/unit/test_db_structure.py`、`tests/unit/test_retrievers_base.py` | 缺少 torch / bert-score / chromadb 時的錯誤訊息 | 很好，但與 public API smoke 還可串接得更完整 |

## 各檔案對照

### `tests/test_akasha.py`

目前是最接近「公開 API 綜合 smoke」的檔案，覆蓋：

- `base_line()` fixture：`akasha.RAG()`
- `test_RAG()`
- `test_ask()`
- `MemoryManager` 的基本查詢

適合對應的矩陣項目：

- `ask()` 最小問答
- `RAG()` 基本 smoke
- `MemoryManager` 基本查詢

缺口：

- 沒有完整 `stream=True` / `thinking=True` / `keep_logs=True` 矩陣
- 沒有多 provider 版本的 smoke
- 沒有 `summary()` / `agents()` 的完整串接

### `tests/test_agent.py`

目前覆蓋：

- `akasha.agents()`
- 自訂 tool
- `keep_logs=True`

適合對應的矩陣項目：

- `agents()` tool calling
- final answer 為 `str`
- logs 行為

缺口：

- 沒有 streaming event assertions
- 沒有 thinking event assertions
- 沒有多 tool / 多 provider smoke

### `tests/test_live_gemini_agent.py`

目前覆蓋：

- 真實 Gemini API
- final answer
- stream + thinking events
- `thinking=False` 時 budget 不應影響

適合對應的矩陣項目：

- `agents()` / `ask()` 的真實 thinking smoke
- streaming contract
- provider-specific behavior

缺口：

- 只有 Gemini
- 缺 OpenAI / Anthropic / Ollama 的真實契約 smoke
- 沒有 tool calling smoke

### `tests/test_summary.py`

目前覆蓋：

- `akasha.summary()`
- `map_reduce`
- `keep_logs=True`
- `chunk_size` / `chunk_overlap` / `max_input_tokens`
- URL 輸入

適合對應的矩陣項目：

- `summary()` 基本 smoke
- URL input
- 參數承接

缺口：

- 缺 text input / file input 的明確 smoke
- 缺 refine 路徑
- 缺回傳內容 schema 的更嚴格檢查

### `tests/test_api_stability.py`

目前覆蓋：

- `ask()` basic API
- `RAG()` smoke
- `agents()` native tool calling
- `vision` smoke
- `MemoryManager` stability
- Windows absolute path regression

適合對應的矩陣項目：

- public API smoke 的主幹
- Windows regression
- 多個高風險路徑的整合驗證

缺口：

- 仍以整合 smoke 為主，缺少 contract 層的細項斷言
- `ask()` 的參數矩陣還不夠密
- `summary()` 的矩陣不完整

## unit 測試對照

### `tests/unit/test_handle_objects_unit.py`

覆蓋：

- model / embedding name 正規化
- `thinking` 設定轉換

對應矩陣：

- provider alias
- `thinking` / `thinking_budget`
- factory 行為

### `tests/unit/test_run_llm_chat_normalization.py`

覆蓋：

- chat input normalization
- reasoning content 分離

對應矩陣：

- `agents()` / `ask()` 的 message contract
- reasoning 不污染 answer

### `tests/unit/test_agents_core.py`

覆蓋：

- agents 核心 alias 與 message 處理

對應矩陣：

- helper / contract
- 非真模型的內部邏輯

### `tests/unit/test_agents_observation_normalization.py`

覆蓋：

- observation / final action 相關格式

對應矩陣：

- agent message/event normalization

### `tests/unit/test_db_structure.py`

覆蓋：

- storage directory sanitize
- Windows absolute path
- URL/path 正規化

對應矩陣：

- `RAG()` / `MemoryManager` 的路徑風險

### `tests/unit/test_retrievers_base.py`

覆蓋：

- torch missing / bert-score missing / chromadb missing
- rerank fallback

對應矩陣：

- light/full 邊界
- provider / dependency contract

## 矩陣項目是否已有保護

| 矩陣項目 | 現況 | 建議 |
|---|---|---|
| `ask()` 最小問答 | 有 | 保留 |
| `ask()` streaming + thinking | 有，但主要是 Gemini live | 補 provider matrix |
| `ask()` logs / env / max_output_tokens | 部分有 | 補 contract assertions |
| `agents()` tool calling | 有 | 保留並補多 tool / 多 provider |
| `agents()` stream + thinking | 有（live Gemini） | 補 OpenAI / Anthropic smoke |
| `RAG()` file / dir smoke | 有 | 保留 |
| `RAG()` Windows path | 有 | 保留為 regression |
| `summary()` text / file / URL matrix | 部分有 | 補 text/file/refine |
| `MemoryManager` persistence | 部分有 | 補重啟 smoke |
| provider alias | unit 有，smoke 不足 | 補真實 provider smoke |
| logs schema | 部分有 | 補 contract test |

## 建議下一步

如果要繼續推進，我建議下一份文件直接寫：

1. `tests/` 新增哪些 smoke 檔
2. 每個 smoke 檔應該放哪些 case
3. 哪些舊測試可以降級為 unit

