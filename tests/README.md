# Akasha 功能導向測試地圖

測試目錄依功能分類；測試速度、外部依賴與升級用途使用 pytest marker 表示，不再使用 `unit/contract/integration/smoke/upgrade_tests` 作為實體測試目錄。

## 功能目錄

```text
tests/
├── ask/          # basic、info/file、info/url、prompt、thinking
├── agent/        # basic、contracts、skills、stream、tools
├── rag/          # input、parameters、retrieval、chroma、provider
├── memory/       # memory 功能測試預留與穩定性測試
├── vision/       # vision provider 測試預留
├── mcp/          # stdio、streamable_http、contracts
├── provider/     # chat、embedding、factory、thinking、compatibility
├── observability/ # logging 與 keep_logs
├── db/、eval/、summary/、compatibility/
├── fixtures/、data/、config/、support/
└── conftest.py
```

`fixtures/` 與 `data/` 是共用測試資源；`support/` 只放路徑 helper 與 coverage utility，不放功能測試。

## 常用指令

```powershell
$env:PYTHONPATH = "."
$python = "C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe"

& $python -m pytest tests/ask -q
& $python -m pytest tests/agent -q
& $python -m pytest tests/rag -q
& $python -m pytest tests/mcp -q
& $python -m pytest tests/memory tests/vision -q
& $python -m pytest -m unit -q
& $python -m pytest --collect-only -q
& $python -m pytest -q -k "not ollama"
```

細功能可以直接執行，例如：

```powershell
& $python -m pytest tests/ask/info/url -q
& $python -m pytest tests/ask/thinking -q
& $python -m pytest tests/rag/parameters -q
& $python -m pytest tests/rag/retrieval -q
& $python -m pytest tests/skill -q
& $python -m pytest tests/mcp/streamable_http -q
```

## Marker 與外部依賴

- `unit`: 不需要 provider 的 deterministic 測試。
- `integration`: 跨模組或完整 runtime 測試。
- `upgrade`: 版本升級與相容性回歸測試。
- `smoke`: 最小 provider/service 驗證；實體位置仍在功能目錄中。
- `requires_api`: 需要 provider API key。
- `full_only`: 需要 full 安裝依賴。

Provider 測試仍由既有 `RUN_*` 環境變數控制，避免一般 pytest 執行意外產生 API 費用：

```powershell
$env:RUN_PROVIDER_SMOKE = "1"
$env:RUN_EMBEDDING_SMOKE = "1"
$env:RUN_RAG_SMOKE = "1"
$env:RUN_RAG_PIPELINE = "1"
$env:RUN_GEMINI_RAG = "1"
$env:RUN_MCP_SMOKE = "1"
$env:RUN_LLM_TESTS = "1"
```

Ollama 測試是例外：只有 Ollama server 可連線時才執行；本次回歸可使用 `-k "not ollama"` 排除。

## 測試資料

- `data/rag/single_fact.txt`: 穩定的 RAG fact。
- `data/rag/directory/`: 目錄與多文件檢索。
- `data/rag/empty.txt`: 空文件邊界。
- `data/rag/unicode_繁中.txt`: 編碼與語言。
- `data/documents/`: PDF、DOCX、JSON 等既有文件。
- `data/images/`: Vision 測試圖片。

新增測試應優先重用既有資料，不應為不同 marker 複製同一份 fixture。

## 覆蓋矩陣

功能覆蓋矩陣與 RAG 參數 audit 位於：

`dev_docs/2026-08-functional-test-coverage-spec.md`

測試通過必須記錄實際命令、passed、skipped、失敗原因，以及是否受到 API key、外部服務或 Ollama 影響。
