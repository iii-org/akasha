# Akasha Package Upgrade Log

Date: 2026-07-06

## Scope

這份文件記錄本次 `akasha-repo` 的依賴調整、測試驗證與已知風險，目的是讓後續升級、回歸調查與封裝同步時有明確參考。

## Objectives

- Python 支援範圍收斂到 `>=3.11,<3.13`
- 以 `pyproject.toml` 作為 single source of truth
- 同步 `requirements.txt` / `requirements-light.txt`
- 拆出 light 與 full 的依賴邊界
- 保留 `light + chromadb + remote embeddings` 的使用方式
- 驗證 Windows `.venv` 下的實際 smoke 路徑

## Dependency Changes

### Python baseline

- `pyproject.toml` 的 `requires-python` 改為 `>=3.11,<3.13`
- README、tests 文件與 CI 指令同步改為 Python 3.11 基線

### requirements synchronization

- 新增並使用 `scripts/sync_requirements.py`
- `requirements-light.txt` 與 `requirements.txt` 改成由 `pyproject.toml` 同步生成
- 移除 `requirements*.txt` 中不應直接維護的手動漂移內容

### light/full boundary

初始判斷曾把 `chromadb` / `langchain-chroma` 從 light 移除，原因是當前 `chromadb` 版本會間接安裝 `onnxruntime`。

後續確認需求後，調整為：

- `light`:
  - 保留 `chromadb`
  - 保留 `langchain-chroma`
  - 使用遠端 embedding API，例如 OpenAI / Gemini
  - 不包含 `torch` / `transformers` / `sentence-transformers`
- `full`:
  - 保留本地模型與重型依賴
  - 包含 `torch` / `torchvision` / `transformers` / `sentence-transformers` / `bert-score` 等

這代表：

- light 仍可做 `RAG` 與 `MemoryManager`
- Chroma 只是向量庫與檢索層
- 向量生成責任改由遠端 embedding provider 承擔

## Code Changes

### Import chain hardening

為了讓 light 安裝不因重型模組在 import 階段被提早載入而失敗，做了以下調整：

- `akasha/__init__.py`
- `akasha/utils/__init__.py`
- `akasha/helper/__init__.py`
- `akasha/utils/db/__init__.py`

改用 lazy import，避免在 package import 時就拉起非必要依賴。

### Text splitter imports

將 `RecursiveCharacterTextSplitter` 改為延遲匯入，避免 collection / coverage 階段經由 `spacy/thinc` 匯入鏈引出不必要風險：

- `akasha/utils/db/create_db.py`
- `akasha/tools/summary.py`

### Retriever fallback

`akasha/utils/search/retrievers/base.py` 做了 fallback 收斂：

- `rerank` 在缺少 `torch` 時給出清楚錯誤
- 不需要 embedding 的檢索路徑不會提早初始化 heavy provider

### Chroma compatibility helper

新增：

- `akasha/utils/db/chroma_compat.py`

用途：

- 將 `chromadb` / `langchain-chroma` 的錯誤訊息集中管理
- 在缺少 Chroma 依賴時提供一致提示

## Windows Absolute Path Fix

### Symptom

Windows `.venv` live smoke 顯示：

- `RAG(data_source="C:\\...\\file.txt")` 失敗
- `MemoryManager(memory_dirname="C:\\...")` 失敗
- Chroma 初始化或文件載入最終出現 `os error 123`

### Root cause

`akasha/utils/db/db_structure.py:get_storage_directory()` 會把 Windows absolute path 的 `C:\...` 直接拼成 `chromadb/...` 目錄名，造成非法字元進入 storage path。

### Fix

- 將 storage path 生成改為逐段 sanitize
- Windows drive / absolute path 會轉成安全片段，例如：
  - `C:\Users\today\Projects\...`
  - 轉為 `chromadb/C-Users-today-Projects-...`

### Verification

- `tests/unit/test_db_structure.py`
  - 新增 Windows absolute path 的 storage directory 單元測試
- Windows live smoke 通過：
  - `OpenAI + Chroma + absolute file path`
  - `MemoryManager + absolute storage path`

## Test Work Performed

### Unit tests

已補或調整：

- `tests/unit/test_db_structure.py`
  - storage dir URL/path handling
  - Windows absolute path sanitization

- `tests/unit/test_retrievers_base.py`
  - rerank fallback
  - custom retriever / search type behavior

### Upgrade/integration tests

已整理或補充：

- `tests/upgrade_tests/test_light_restrictions.py`
  - 缺少 `torch/transformers`
  - 缺少 `bert-score`
  - 缺少 `chromadb/langchain-chroma`

- `tests/test_api_stability.py`
  - `ask`
  - `RAG`
  - `agent`
  - `vision`
  - `MemoryManager`
  - Windows absolute path smoke

### Live smoke executed during this task

在使用者提供的 Windows `.venv` 與 API keys 下，已實測：

- `OpenAI + Chroma + openai:text-embedding-3-small`
- `Gemini + Chroma + google:gemini-embedding-001`
- `MemoryManager + Chroma + openai:text-embedding-3-small`
- Windows absolute path regression after fix

## Known Constraints

- `chromadb` 目前仍可能間接安裝 `onnxruntime`
- 這不代表 light 會使用本地 embedding，只代表套件依賴鏈較重
- Windows `pytest-cov` 仍可能因 `spacy/thinc/torch` collection 鏈而不穩定
- `tests/upgrade_tests/*` 若直接跑 pytest，有些案例不會自動載入 `tests/.env`，需留意啟動方式

## Recommended Commands

### Sync requirements

```bash
python scripts/sync_requirements.py
```

### Unit baseline

```bash
../.venv/Scripts/python.exe -m pytest tests/unit -m unit
```

### Light smoke

```bash
../.venv/Scripts/python.exe -m pytest tests/test_api_stability.py -m "upgrade and requires_api" -s
```

### Windows absolute path regression

```bash
../.venv/Scripts/python.exe -m pytest tests/test_api_stability.py -k "windows_absolute_path" -s
```

## Follow-up Suggestions

- 若後續繼續升級 `chromadb`，優先重跑：
  - `tests/unit/test_db_structure.py`
  - `tests/test_api_stability.py -k "windows_absolute_path"`
  - `tests/test_api_stability.py::test_rag_smoke`
  - `tests/test_api_stability.py::test_memory_stability`

- 若要再收斂 light 安裝體積，可後續評估：
  - 是否存在不拉 `onnxruntime` 的 `chromadb` 版本
  - 或是否要提供替代 vector store 方案
