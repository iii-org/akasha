# Akasha Testing Guide

這份文件記錄本次對 `akasha-repo` 測試體系的整理方式，供後續升級與功能回歸時直接沿用。

## 目標

- `Light` 安裝形態優先。
- `unit` 測試作為 coverage 主體，不依賴 API key、GPU 或地端大模型。
- `upgrade` / `integration` 測試保留真實 smoke 路徑，用來比較升級前後行為是否改變。

## 測試分層

- `tests/unit/`
  - 快速單元測試。
  - 以 mock、fake object、monkeypatch 為主。
  - 目前覆蓋重點：
    - `akasha/utils/db/db_structure.py`
    - `akasha/utils/search/retrievers/base.py`
    - `akasha/helper/encoding.py`
    - `akasha/helper/crawler.py`
    - `akasha/helper/base.py`
    - `akasha/helper/preprocess_prompts.py`
    - `akasha/utils/search/search_doc.py`
    - `akasha/agent/agents.py` 的核心 alias 行為

- `tests/upgrade_tests/`
  - 真實 provider smoke 測試。
  - 適合依賴升級前後比較。
  - 需要 `.env` 與 API keys。

- 既有 `tests/test_*.py`
  - 仍保留，但已補上 `integration` / `requires_api` / `smoke` 等 marker。
  - 這些測試不應作為 coverage fail-under 主體。
  - Windows 特有的 absolute-path 回歸測試也收斂在這一層。

## Marker 約定

- `unit`: 不依賴外部 API 的快速測試
- `integration`: 真實整合測試
- `upgrade`: 升級前後要重跑的 smoke 測試
- `requires_api`: 需要 provider API key
- `full_only`: 需要 full 安裝與本地模型依賴
- `smoke`: 最小可行端到端驗證

## Coverage Policy

coverage 設定已放進 `pyproject.toml`。

- 納入範圍：
  - `akasha/helper`
  - `akasha/utils/db`
  - `akasha/utils/search`
  - `akasha/tools/ask.py`
  - `akasha/tools/summary.py`
  - `akasha/agent`
  - `akasha/RAG`

- 排除範圍：
  - `akasha/ui.py`
  - `akasha/interface/*`
  - `akasha/api.py`
  - `akasha/utils/models/*`
  - `akasha/tools/gen_img.py`

## 建議指令

建議先以 Python 3.11 建立 light 環境，例如 `uv venv --python 3.11` 後再執行以下命令。

如果 light 環境位於 `akasha-update/.venv`：

```bash
../.venv/Scripts/python.exe -m pytest tests/unit -m unit
```

量測核心 coverage：

```bash
../.venv/Scripts/python.exe -m pytest tests/unit -m unit --cov --cov-report=term-missing
```

如果 `pytest-cov` 在目前的 `light` 環境中因 `spacy/thinc/torch` 匯入鏈而不穩定，可改用 repo 內建的 coverage harness：

```bash
../.venv/Scripts/python.exe tests/measure_unit_coverage.py
```

這個腳本會直接執行目前的核心 `unit` 測試邏輯，並計算以下模組的整體 line coverage：

- `akasha/agent/agents.py`
- `akasha/helper/base.py`
- `akasha/helper/crawler.py`
- `akasha/helper/encoding.py`
- `akasha/helper/preprocess_prompts.py`
- `akasha/utils/db/db_structure.py`
- `akasha/utils/logging_config.py`
- `akasha/utils/search/retrievers/base.py`
- `akasha/utils/search/search_doc.py`

跑升級 smoke：

```bash
../.venv/Scripts/python.exe -m pytest tests/upgrade_tests -m "upgrade and requires_api" -s
```

如果要針對 Windows absolute path 問題做回歸驗證，建議在 Windows `.venv` 直接跑：

```bash
../.venv/Scripts/python.exe -m pytest tests/test_api_stability.py -k "windows_absolute_path" -s
```

這兩個案例會驗證：

- `RAG(data_source=<Windows absolute file path>)`
- `MemoryManager(memory_dirname=<Windows absolute path>)`

## 升級前後建議流程

1. 在現有版本先跑 `tests/unit` 與 `tests/upgrade_tests`。
2. 升級依賴。
3. 重跑相同命令。
4. 比對：
   - coverage 是否仍達門檻
   - `ask` / `RAG` / `summary` / `agent` / `memory` smoke 是否仍可通過
   - retriever fallback、logging、memory 等非模型品質邏輯是否有回歸
   - Windows absolute path 是否仍可正常建立與讀取 Chroma 儲存目錄

## 注意事項

- 若未來要把 80% 門檻擴張到更多模組，優先擴到 `tool-flow` 與 `RAG` 控制邏輯，再考慮 provider adapter。
- `full_only` 路徑目前仍應獨立維護，不要和 light coverage KPI 綁在一起。
- 目前 Windows `light` venv 直接跑 `pytest-cov` 時，可能觸發 `langchain_text_splitters -> spacy -> thinc -> torch` 的匯入鏈，進而在 collection/coverage 階段出現不穩定或 runtime error。遇到這種情況時，優先使用 `tests/measure_unit_coverage.py` 取得可重現的 coverage 數字。
- 2026-07 這次依賴調整曾發現 Windows absolute path 會在 Chroma storage path 生成時觸發 `os error 123`。目前已由 `tests/unit/test_db_structure.py` 與 `tests/test_api_stability.py -k windows_absolute_path` 共同覆蓋，後續若再動 `get_storage_directory()`，這兩層測試都應重跑。
