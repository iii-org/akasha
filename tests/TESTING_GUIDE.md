# Akasha Testing Guide

本專案測試以功能目錄為主，pytest marker 為輔。請依照要驗證的功能選擇路徑，不要依 `unit`、`smoke` 或 `integration` 尋找測試檔案。

## 功能路徑

| 功能 | 路徑 |
|---|---|
| Ask | `tests/ask/` |
| Agent | `tests/agent/` |
| RAG | `tests/rag/` |
| Memory | `tests/memory/` |
| Vision | `tests/vision/` |
| MCP | `tests/mcp/` |
| Skills | `tests/skill/` |
| Tools | `tests/tool/` |
| Provider | `tests/provider/` |
| Observability | `tests/observability/` |

## 分類例子

```text
tests/ask/stream/
tests/ask/info/file/
tests/ask/info/directory/
tests/ask/info/url/
tests/ask/thinking/
tests/rag/parameters/
tests/rag/retrieval/
tests/rag/chroma/
tests/skill/
tests/mcp/streamable_http/
```

## Marker

測試檔案可以同時具有多個 marker：

```python
pytestmark = [pytest.mark.integration, pytest.mark.requires_api]
```

marker 只描述測試成本與依賴，不改變功能目錄位置。常用 marker 包含 `unit`、`integration`、`upgrade`、`smoke`、`requires_api` 與 `full_only`。

## 執行

```powershell
$env:PYTHONPATH = "."
$python = "C:\Users\today\Projects\akasha-update\.venv\Scripts\python.exe"

& $python -m pytest --collect-only -q
& $python -m pytest tests/ask -q
& $python -m pytest tests/rag -q
& $python -m pytest tests/agent tests/mcp -q
& $python -m pytest -m unit -q
& $python -m pytest -q -k "not ollama"
```

Provider live tests必須明確設定對應的 `RUN_*` 開關與 API key。Ollama server 未啟動時，Ollama 案例屬於明確例外，可以使用 `-k "not ollama"` 排除。

## 驗證規則

完成測試整理後至少執行：

```powershell
& $python -m pytest --collect-only -q
& $python -m py_compile <changed-test-files>
& $python -m pytest -q -k "not ollama"
```

報告中要區分：測試通過、環境 skip、Ollama 例外、provider/API 問題、以及 Akasha 本身的程式錯誤。
