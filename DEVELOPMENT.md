# Development Guide

本文件說明專案工具細節、主要功能與簡單範例、主要資料夾/程式檔用途，以及本次升級的變更記錄。

## 專案簡介
`akasha_light` 是以 LangChain 為基礎的文件 QA / RAG 與 Agent 工具組，支援多模型（包含 Gemini），並提供資料建庫、檢索、摘要、評估與 MCP 整合。

## 主要功能與簡單範例

### 1) RAG（文件問答）
```python
import akasha

ak = akasha.RAG(
    embeddings="gemini:gemini-embedding-001",
    model="gemini:gemini-2.5-flash",
    env_file="tests/test_upgrade/.env",
)
print(ak("./docs/mic", "五軸是甚麼?"))
```

### 2) Ask（快速問答）
```python
import akasha

ak = akasha.ask(
    model="gemini:gemini-2.5-flash",
    env_file="tests/test_upgrade/.env",
)
print(ak("chromadb 版本？", ["chromadb==0.4.14"]))
```

### 3) Summary（摘要）
```python
import akasha

summ = akasha.summary(
    "gemini:gemini-2.5-flash",
    env_file="tests/test_upgrade/.env",
)
print(summ(content=["https://github.com/iii-org/akasha"]))
```

### 4) Agent（工具與記憶）
```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    env_file="tests/test_upgrade/.env",
    verbose=True,
)
print(agent("今天幾月幾號？"))
```

### 5) MCP（Model Context Protocol）
`tests/test_upgrade` 內提供最小 MCP server + client：
```bash
python tests/test_upgrade/mcp_client_example.py
```

## 主要資料夾與程式檔

### 核心程式
- `akasha/`：核心套件
  - `akasha/RAG/`：RAG 相關流程
  - `akasha/agent/`：Agent 相關邏輯、工具整合
  - `akasha/tools/`：ask/summary/websearch 等高階工具
  - `akasha/utils/`：模型封裝、資料庫、檢索器、prompt 等工具
  - `akasha/eval/`：評估流程與評分
  - `akasha/helper/`：共用功能（模型處理、記憶、切詞等）

### 其他
- `tests/`：pytest 測試
- `examples/`：範例腳本
- `docs/`：說明與文件
- `tests/test_upgrade/`：升級驗證用 MCP 範例與 `.env`

## 環境設定
Gemini API key 放在 `tests/test_upgrade/.env`：
```
GEMINI_API_KEY=your_key_here
```

## 本地開發流程（uv/venv）
1. 建立並啟動 venv（依你的實際路徑調整）：
```bash
source /Users/today/Projects/Envs/akasha-upgrade/.venv/bin/activate
```
2. 安裝依賴（建議用 uv）：
```bash
uv pip install -e .
```
3. 執行測試：
```bash
pytest -vv -m "akasha or agent or helper"
```

## MCP 測試
1. 確保 `tests/test_upgrade/.env` 已設定 `GEMINI_API_KEY`。
2. 執行 MCP client 範例（會自動啟動 stdio MCP server）：
```bash
python tests/test_upgrade/mcp_client_example.py
```
3. 單獨跑 MCP 測試：
```bash
pytest -vv tests/test_mcp_upgrade.py
```

## 發版流程（GitHub → PyPI）

### 版本號更新
1. 更新版本號：
   - `pyproject.toml` 的 `project.version`
   - 如果有 `setup.py` 或其他版本檔，同步更新
2. 確認測試：
```bash
pytest -vv -m "akasha or agent or helper"
```
3. 提交並推到 `akasha-light`：
```bash
git add -A
git commit -m "chore: bump version to X.Y.Z"
git push origin akasha-light
```

### PR 合併到 master
1. 建立 PR：`akasha-light` → `master`
2. CI 會在 `akasha-light` push 時跑測試（見 `.github/workflows/ci.yml`）
3. 合併 PR 後，`master` 的 `publish.yml` 會 build + twine check（不會發佈）

### 發佈到 PyPI（tag 觸發）
1. 建立 tag 並推送：
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```
2. `publish.yml` 在 `refs/tags/v*` 時會執行 `twine upload`
3. 成功後 PyPI 會出現新版本

### 注意事項
- `publish.yml` 只有在 tag push 才會發佈到 PyPI
- `master` push 只會 build / twine check，不會 upload
- 需要 GitHub Secrets：`PYPI_API_TOKEN` 及 CI 需要的 API key（如 `GEMINI_API_KEY`）

## 版本 1.0 升級 Change Log（LangChain 1.x 路線）

### 依賴升級
- LangChain 生態系升級至 1.x：
  - `langchain-core>=1.0,<2.0`
  - `langchain>=1.0,<2.0`
  - `langchain-community>=0.4.1,<0.5`
  - `langchain-classic>=1.0,<2.0`
  - `langchain-openai>=1.0,<2.0`
  - `langchain-text-splitters>=1.1,<2.0`
  - `langchain-chroma>=1.1,<2.0`
  - `langchain-mcp-adapters>=0.2,<0.3`
- `chromadb` 升級至 1.x 分支（由 `langchain-chroma` 連動）。

### 主要程式碼調整
- `langchain.schema` / `langchain.docstore` / `langchain.tools`
  → 改為 `langchain_core.*`
- `langchain.chains.query_constructor...`
  → 改為 `langchain_classic.chains...`
- `langchain.text_splitter`
  → 改為 `langchain_text_splitters`
- `langchain.llms.base.LLM`
  → 改為 `langchain_core.language_models.LLM`
- `langchain.embeddings`
  → 改為 `langchain_community.embeddings`

### 測試
執行：
```bash
pytest -vv -m "akasha or agent or helper"
```
結果：`6 passed, 5 deselected`
