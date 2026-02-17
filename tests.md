這個文件的目的是規劃 Akasha 套件的測試個案。為了適應不同的執行環境，我們將測試分為 **[Light]** (輕量級/API版) 與 **[Full]** (全功能/環境版)。

---

## 1. [Light] 版測試範圍 (Lightweight Scope)
此版本專注於邏輯與遠端 API，不需強大算力或 GPU。

### 1.1 遠端模型與基礎問答
- [ ] **OpenAI/Gemini/Anthropic API**: 驗證基本 `ask` 與 `RAG` 呼叫。
- [ ] **基礎摘要 (Summary)**: 使用 API 模型進行 `map_reduce` 與 `refine` 摘要。
- [ ] **記憶管理 (Memory)**: 驗證對話上下文與摘要存儲邏輯。
- [ ] **基礎 Agent**: 使用內建的 `websearch_tool` 與自定義簡單 Python 工具。

### 1.2 核心邏輯與數據處理
- [ ] **多格式文件讀取**: 測試 `.txt`, `.pdf` 基礎解析。
- [ ] **Token 計算**: 驗證 API 模型的 Token 估算精準度。
- [ ] **API 結構化輸出**: 確保回傳格式符合程式碼解析需求。

### 1.3 輕量檢索器
- [ ] **基礎檢索**: 測試 `tfidf`, `knn` 等純運算不需要大模型的檢索方式。

---

## 2. [Full] 版測試範圍 (Full Functionality Scope)
此版本包含所有功能，通常需要 GPU 與較大的硬碟空間。

### 2.1 地端模型與量化
- [ ] **Llama-cpp (GGUF)**: 於本地 CPU/GPU 執行量化模型。
- [ ] **HuggingFace (HF/GPTQ)**: 測試 Transformer 模型載入與推理。

### 2.2 多模態與整合工具
- [ ] **影像處理 (Vision)**: 測試 `gen_image`, `edit_image` 以及多模態 VQA。
- [ ] **進階檢索 (Retrievers)**: 測試 `rerank` (如 Bge-reranker) 與大型向量數據庫 (ChromaDB) 的深度整合。
- [ ] **MCP 工具整合**: 與地端啟動的 MCP 伺服器進行互動測試。

### 2.3 性能與壓力測試
- [ ] **超長文本摘要**: 測試單次處理數百頁文檔的穩定性。
- [ ] **並發調用**: 測試多個 Agent 或 RAG 同時執行的資源消耗。

---

## 3. 功能詳細對應表 (Detailed Test Cases)

### 1.1 RAG 核心
- [ ] **初始化測試**: 驗證不同的模型、嵌入模型、chunk 大小、搜尋類型是否能正確初始化。
- [ ] **基本問答 (Ask)**: 給予一段文件，驗證模型是否能根據文件內容回答問題。
- [ ] **多種數據源支援**: 測試內容來源為單一目錄、多重目錄、檔案清單、或是 ChromaDB 名稱。
- [ ] **搜尋類型測試**: 測試 `auto`, `svm`, `tfidf`, `rerank` 等不同的 search types。
- [ ] **Self-Ask RAG**: 驗證對複雜問題的拆解與回答能力。
- [ ] **引用來源 (Reference)**: 驗證回答後是否能正確返還參考的文件來源。

### 1.2 摘要功能 (Summary)
- [ ] **Map-Reduce 模式**: 測試長文本的摘要流程。
- [ ] **Refine 模式**: 測試循序漸進的摘要一致性。
- [ ] **多種輸入媒介**: 測試 URL (網頁內容)、本機檔案 (.pdf, .docx, .txt) 以及純文字輸入。
- [ ] **語言測試**: 測試中文與英文的摘要輸出品質。

### 1.3 代理人功能 (Agents)
- [ ] **內建工具整合**: 測試 `websearch_tool` 與 `saveJSON_tool`。
- [ ] **自定義工具**: 測試透過 `create_tool` 建立的工具是否能被 Agent 正確識別並執行。
- [ ] **MCP 工具整合**: 驗證與 MCP 伺服器的連接與工具調用。
- [ ] **多輪對話 (Memory)**: 驗證 Agent 是否能記住先前的對話內容並做出反應。
- [ ] **串流輸出 (Stream)**: 驗證 Agent 的串流回應模式。

### 1.4 評估功能 (Eval)
- [ ] **題庫生成 (Create Questionset)**: 驗證根據文件自動生成問答集的功能。
- [ ] **自動評分**: 測試使用 BERTScore 或其他模型對回答進行評估。

### 1.5 影像處理與多模態 (Vision & Multimodal)
- [ ] **影像生成 (gen_image)**: 驗證模型是否能根據提示詞生成圖像。
- [ ] **影像編輯 (edit_image)**: 驗證基礎影像修改功能。
- [ ] **多模態問答**: 測試模型是否能理解影像內容並回答問題 (VQA)。

### 1.6 記憶管理 (Memory Management)
- [ ] **記憶儲存與檢索**: 驗證 `MemoryManager` 是否能正確將對話摘要存入數據庫。
- [ ] **對話上下文**: 測試在長對話中，記憶是否能有效被提取並用於生成回應。

### 1.7 檢索器測試 (Retrievers)
- [ ] **多種檢索算法**: 測試 `bm25`, `svm`, `tfidf`, `knn`, `mmr` 的檢索結果。
- [ ] **Rerank 整合**: 驗證 `rerank:BAAI/bge-reranker-base` 整合後的準確度提升。
- [ ] **自定義檢索器**: 測試用戶定義的檢索函數。

## 2. API 與介面測試 (API & Interface Tests)

### 2.1 FastAPI 介面
- [ ] **RAG 端點**: 測試 `/RAG` 路由的 POST 請求。
- [ ] **Ask 端點**: 測試 `/ask` 路由。
- [ ] **Summary 端點**: 測試 `/summary` 路由。
- [ ] **Websearch 端點**: 測試 `/websearch` 路由。

### 2.2 CLI 介面
- [ ] **終端交互**: 測試 `akasha_terminal` 的啟動與基本命令。

## 3. 系統與公用工具測試 (System & Utility Tests)

### 2.1 Token 計算與限制
- [ ] **Tokenizer**: 驗證不同模型的 Token 計算精準度。
- [ ] **Max Input Tokens**: 當輸入超過限制時，系統是否能正確處理 (例如截斷或拋出異常)。

### 2.2 日誌紀錄 (Logging)
- [ ] **日誌保存**: 驗證 `save_logs` 是否能正確產出 JSON 或 TXT 格式日誌。
- [ ] **日誌配置**: 驗證手動與自動配置日誌的行為。

### 2.3 環境配置
- [ ] **.env 載入**: 驗證 API Key 與設定是否能從不同的 .env 路徑載入。

## 3. 整合與邊際情況測試 (Integration & Edge Cases)

### 3.1 跨模型驗證
- [ ] **OpenAI/Azure OpenAI**
- [ ] **Google Gemini**
- [ ] **Anthropic Claude**
- [ ] **HuggingFace (HF/GPTQ)**
- [ ] **Llama-cpp (GGUF)**

## 4. 模型相容性與穩定性測試 (Model Compatibility & Stability)

### 4.1 模型接口穩定性 (API Shield)
- [ ] **結構化輸出驗證**: 針對 OpenAI, Anthropic, Gemini 等模型，驗證輸出是否符合預期的 JSON 或特定格式，防止 API 更新導致解析失敗。
- [ ] **Provider 健康檢查**: 模擬 API 回傳錯誤，驗證系統的重試機制或降級策略。

### 4.2 遠端與地端模型切換
- [ ] **Remote APIs**: OpenAI (GPT-4o), Gemini (Flash/Pro), Anthropic (Claude-3.5)。
- [ ] **Local Models**: 測試 `llama-cpp` (GGUF) 於 CPU 運行，以及 `hf`/`gptq` 於 GPU 運行。
- [ ] **參數一致性**: 確保不同模型在相同 `temperature` 與 `max_tokens` 設定下的行為大致符合預期。

## 5. 測試工具與環境架設 (Tools & Setup)

### 5.1 測試工具建議
- 使用 `pytest` 作為主要測試框架。
- 使用 `pytest-asyncio` 測試非同步功能 (如 Agent)。
- 使用 `pydantic` 進行參數校驗 (如果需要)。
- 利用 `tests/upgrade_tests/` 下的結構來進行跨平臺相容性測試。

### 5.2 環境架設建議 (使用 `uv`)
為了確保測試環境與開發代碼同步，推薦使用以下流程：

1. **建立虛擬環境**:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate  # 或您的環境路徑
   ```

2. **開發模式安裝 (Editable Install)**:
   使用此模式安裝可確保 `import akasha` 永遠指向您目前的原始碼目錄，且具備開發與測試所需的工具：
   ```bash
   # [Light] 版開發安裝
   uv pip install -e ".[light,dev]"

   # [Full] 版開發安裝
   uv pip install -e ".[full,dev]"
   ```

3. **驗證安裝路徑**:
   執行以下指令確認系統抓到的是您 Dropbox 目錄下的 `akasha`：
   ```bash
   python -c "import akasha; print(akasha.__file__)"
   ```

### 5.3 執行測試
```bash
# 執行 API 穩定性測試
pytest tests/test_api_stability.py -s
```

---

## 6. 測試準備清單 (Preparation Checklist)

為了確保測試能順利執行，請根據您要進行的測試版本準備以下資源：

### 6.1 [Light] 版準備事項
**適合一般 API 開發環境，僅需配置 API Key 與基礎文件。**

- **API 金鑰 (.env)**:
  - [ ] `OPENAI_API_KEY`: 必備，用於多數預設功能。
  - [ ] `GEMINI_API_KEY`: 用於測試 Google 系列模型。
  - [ ] `ANTHROPIC_API_KEY`: 用於測試 Claude 系列模型。
  - [ ] `BRAVE_API_KEY` 或 `SERPER_API_KEY`: 用於測試 `websearch` 功能。
- **測試數據 (test_data/)**:
  - [ ] **基礎文件**: 1-5 頁的 `.txt` 或 `.pdf` (RAG 基本測試)。
  - [ ] **結構化數據**: 一個簡單的 `.json` 檔案。
- **環境設定**:
- [ ] 安裝 `requirements-light.txt` 內的依賴（已包含 `bert-score` 以支援 eval 測試）。

### 6.2 [Full] 版準備事項
**需要 GPU 算力、較大磁碟空間，包含所有本地推理功能。**

- **擴充金鑰 (.env)**:
  - [ ] 所有 [Light] 版的金鑰。
  - [ ] `AZURE_API_KEY`: 測試 Azure OpenAI 特定整合。
  - [ ] `HUGGINGFACEHUB_API_TOKEN`: 用於下載門檻較高的權重檔。
- **測試數據 (test_data/)**:
  - [ ] **深度文件**: 超過 20 頁的 `.pdf` 或 `.docx` (壓力測試)。
  - [ ] **多模態素材**: 影像檔 `.jpg`, `.png` (Vision 測試)。
- **環境與模型**:
  - [ ] 安裝 `requirements.txt` (包含 torch, llama-cpp-python 等重型依賴)。
  - [ ] **地端模型**: 準備 `.gguf` 或 `GPTQ` 模型檔，並記下存放路徑。
  - [ ] **向量庫設定**: 確保具備磁碟寫入權限，用於 ChromaDB 數據持久化測試。

---
---

## 7. 已知問題與環境紀錄 (Known Issues & Environment Logs)

### 7.1 版本相依性與衝突紀錄
在進行 [Light] 版穩定性測試時，發現以下重要的版本配套需求：

- **ChromaDB 與 PostHog 斷層**:
  - **現象**: 使用 `chromadb >= 0.5.x` 搭配最新版 `posthog 7.4.x` 時，會噴出 `capture() takes 1 positional argument but 3 were given` 的錯誤。
  - **原因**: PostHog SDK 6.0 之後更改了簽名，但 ChromaDB 尚未跟進。
  - **解決方案**: 鎖定 `posthog < 6.0.0` 並升級 `chromadb >= 0.6.3` (或穩定版 0.5.18)。
- **Numpy 相容性**:
  - **現象**: `chromadb 0.6.x` 回傳格式改為 `numpy.ndarray`，導致舊有 `akasha` 代碼中的 `.append()` 失敗。
  - **已修復**: `akasha/utils/db/db_structure.py` 已加入強制 `tolist()` 轉換。

### 7.2 已知警告 (Known Warnings) - 持續追蹤中
為了維持系統透明度，以下訊號目前**特意保留**不予遮蔽：

- **Pydantic 棄用警告 (Deprecation)**:
  - **訊息**: `PydanticDeprecatedSince211: Accessing the 'model_fields' attribute...`
  - **來源**: `chromadb 0.6.3` 內部調用。
  - **影響**: 無功能影響，需等待 ChromaDB 官方更新至 Pydantic 3.0 相容語法。
- **PDF 結構警告 (pypdf)**:
  - **訊息**: `Ignoring wrong pointing object...`
  - **來源**: 當輸入的 PDF 檔案結構非標準或曾經過度壓縮時，`pypdf` 負載器會噴出此警告。
  - **監控方式**: 已在 `logging_config.py` 中將 `pypdf` 設為 `ERROR` 等級以減少主控台噪音，但底層仍會運作。

### 7.3 已修正之缺陷 (Fixed Bugs)
- **MemoryManager env_file 支援**: 修正了 `MemoryManager` 初期無法接收自定義 `.env` 路徑的問題，現已與 `RAG`、`ask` 保持一致。

---

## 8. 測試進度紀錄 (Test Progress)
- [x] **[Light] API 穩定性基礎測試**: 通過 (OpenAI/Gemini 交叉驗證)。
- [x] **Memory 基礎存儲測試**: 通過。
- [x] **Agent JSON 格式驗證**: 通過。
- [ ] **多格式文件讀取壓力測試**: 待執行。
- [ ] **Rerank 準確度基準測試**: 待執行。
```
