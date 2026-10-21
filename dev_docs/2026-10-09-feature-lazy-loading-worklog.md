# 全功能延遲載入：第一輪實作與驗證紀錄

日期：2026-10-09

依據：[PRD](2026-10-09-feature-lazy-loading-prd.md)

## 結論

本輪修改涵蓋全版共用程式碼與選配功能入口，不維護另一份 light 實作。測試先使用目前的基礎／light 環境，沒有安裝或升級 full 套件。

不必要的功能載入已減少，公開使用方式、LangChain Agent graph、串流與 Skills REPL 的既有測試通過。但真實問答複測沒有穩定重現初輪加速，不能宣稱約 30 秒的首次啟動問題已解決。昂貴中文資源由建構移至第一次實際使用，必須看首次取得答案的整段時間。

## 1. 模組責任與共同寫法

| 模式 | 實作位置 | 規則 |
| --- | --- | --- |
| 公開匯出 | `agent/__init__.py`、`agent/skills/__init__.py` | 沿用明確匯出表與 `__getattr__`，不建立通用動態 loader |
| 功能選擇 | DB、文件、retriever、ROUGE、圖片預覽、Skills middleware／REPL | 在真正選定的分支明確 import；Python 自行重用已匯入模組 |
| 資源建立 | `helper/chinese.py` | `get_tokenizer()`／`get_converter()` 以鎖保護首次建立，成功後重用，失敗不寫入快取 |

Agent facade 與 `agents.py` 同名，因此保留輕量類別匯出，避免直接 import 子模組後取得錯誤的模組物件。`agents.py` 中的小型 `create_agent()` 入口直到建構時才載入 LangChain graph factory，並維持既有 patch 位置。未選 Skills 時不載入 middleware，未執行 REPL 時不載入 experimental REPL。

`helper/base.py` 保留原 helper 函式與 `cc`／`jieba_tokenizer` 的相容存取；昂貴初始化集中於 Chinese 資源模組。本輪不全面搬動所有純工具，以控制重構範圍。沒有透明代理物件、全域 client singleton 或依賴版本變更。

## 2. 已修改內容

- 共用基底不預先載入 DB；取得 `dbs` 物件時也不載入 DB 建置實作。
- 純文字與 Document 輸入不載入檔案／網頁處理器；檔案、目錄、URL 分支各自解析。
- URL 載入才載入 crawler，保留 title／url metadata。
- 名稱 helper 與英文長度不初始化中文資源；簡繁轉換與中文斷詞首次使用才建立。
- Agent／Skills 匯出分離；工具與 registry 不載入 graph／middleware／REPL。
- 檢索選擇分支只載入需要的策略，保留 auto／merge 的原組合。
- LLM 評分不載入 ROUGE；英文 ROUGE 不載入中文斷詞。選定 ROUGE 缺套件時明確拋出 ImportError，不吞成 0 分。
- 圖片匯出不載入 client；圖片預覽才載入 Pillow 預覽實作。Pillow 仍是必要 dependency，沒有從 packaging 移除。
- API cleanup 不首次 import Torch，只清理已載入且 CUDA 可用的資源。
- 原本已按需載入的 provider、FAISS、MLflow、本地模型等入口保持原設計，不為形式一致而重寫。

既有測試中，DB／文件／圖片 client／retriever 的部分 patch 位置改為實作所屬模組；原有回傳值、參數與錯誤斷言保留。

## 3. TDD 與回歸

新增 `tests/provider/factory/test_feature_loading.py`，共 18 項測試。

新增載入行為按切片先執行失敗案例，再修改實作至通過；另補充既有功能相容性斷言。涵蓋名稱 helper、英文長度、純文字／Document、Ask 建構、工具實際 invoke、registry 身分、無 Skills 的本地 Agent 執行、middleware／REPL、BM25、評分、圖片匯出、API cleanup、中文初始化併發／失敗重試、URL metadata 與 Agent 匯入順序。

最終驗證：

| 執行範圍 | 結果 |
| --- | --- |
| `not live and not full_only` | 312 passed、1 skipped、62 deselected |
| 新增 feature loading 測試 | 18 passed，已包含在上述回歸中，不重複計算 |
| Gemini Ask 與 Agent 真實 smoke | 4 passed，含非串流、串流／thinking 與 thinking 關閉 |
| 兩輪受控 API 時間比較 | 24 次問答全部回傳預期 `ASK_SMOKE_OK` |
| scoped `git diff --check` | 通過；僅有既有 Windows 行尾提示 |

略過案例是 Windows 符號連結權限，不是 full 已通過的證據。Full／GPU／UI／MLflow／FAISS 等真實 runtime 尚未驗證；CUDA 清理測試使用替身，不宣稱真實 GPU 成功。

本機 repo 下的 pytest 暫存目錄曾造成 fixture 權限錯誤，改用唯一的 `C:/tmp/akasha-lazy-*` 暫存根後通過。沒有修改測試政策或產品行為來掩蓋此問題。既有 MCP、langchain-experimental、aiohttp 的 deprecation warnings 保留，不在本輪升級或替換依賴。

## 4. 效能方法

- 修改前 source snapshot：`cb65a1df57821235d25bc65126a6d2495c06b783` 的 `akasha/`，由 Git archive 取出，未覆寫目前 checkout。
- 比較腳本：`scripts/benchmark_feature_startup.py`。
- 同一 Python 3.11.11、同一 `.venv`、同一 `tests/.env`、同一 Gemini 2.5 Flash。
- 同一問句，`language='ch'`、`max_output_tokens=128`、`thinking_budget=0`。
- 每輪三組，before／after 順序輪替，所有 benchmark child 循序執行。
- Cold 是新程序＋各自空 bytecode cache；warm 是另開程序並重用該 cold 的快取。沒有清空 OS／磁碟快取。
- 每個 child 實際呼叫一次 Gemini，不輸出金鑰或模型內容，只存成功、字串符合與時間。
- 未啟用 WPR／ETW／cProfile／import hooks。時間不包含 interpreter 與診斷 harness 啟動。
- `first_call` 含首次中文資源初始化、Akasha 流程、連線與模型延遲，不是純 API 延遲。`first_total` 是 import＋建構＋第一次問答，每次先相加再取中位數。
- 測試先採 light 環境；不將本機結果當作 full／乾淨機器／跨機器的保證。

## 5. 時間結果

### 初輪：主要載入修改完成後

| 條件與階段 | 修改前中位數 | 修改後中位數 | 差異 |
| --- | --- | --- | --- |
| Cold 首次就緒 | 24.84 s | 20.81 s | 減少 4.03 s |
| Cold 首次取得答案 | 26.92 s | 24.04 s | 減少 2.88 s |
| Warm 首次就緒 | 5.22 s | 3.95 s | 減少 1.28 s |
| Warm 首次取得答案 | 7.50 s | 6.97 s | 減少 0.53 s |

此時初步有加速，但後續仍補上 Agent 匯出相容性、annotation 整理與評分缺依賴處理，不能只引用初輪作為最終結論。

### 最終版複測

| 條件與階段 | 修改前中位數 | 修改後中位數 | 差異 |
| --- | --- | --- | --- |
| Cold 首次就緒 | 24.45 s | 23.48 s | 減少 0.97 s |
| Cold 首次取得答案 | 26.74 s | 26.75 s | 增加約 0.01 s，幾乎不變 |
| Warm 首次就緒 | 5.14 s | 5.01 s | 減少 0.13 s |
| Warm 首次取得答案 | 6.83 s | 7.66 s | 增加約 0.83 s |

最終版 Cold 首次答案 before 範圍 25.99–28.59 s，after 範圍 25.34–27.49 s；Warm before 6.15–7.16 s，after 6.50–8.56 s。範圍重疊，樣本亦少。Cold 的 `first_call` 中位數由 2.28 s 到 3.27 s，與資源初始化延後的方向一致，但不能將整個差值都歸因於中文資源，因為它也包含網路／模型成本。

兩輪完整問答後的模組總數均為 before 1304、after 1261，少 43 個；不再載入 crawler、file_loader、load_db、create_db、delete_db、extract_db、chroma_compat 等不相關功能。這是載入範圍改善的證據，不是固定秒數加速的證據。

**最終結論：目前沒有證明穩定的端到端加速。** 不應以初輪結果宣稱已改善 10% 或已消除三十秒問題；也不能因這輪較慢就斷言程式必然退步。需要更大樣本與分段追蹤，才能區分 I/O 波動、首次資源成本與網路／模型成本。

## 6. 複現與證據

已有本機 baseline snapshot 時，可使用新的輸出目錄複現；`--live` 會產生 API 費用：

```powershell
.venv/Scripts/python.exe scripts/benchmark_feature_startup.py `
  --before artifacts/startup-profile/lazy-baseline-source `
  --output artifacts/startup-profile/lazy-new-comparison `
  --live --repeats 3
```

腳本拒絕重用已存在的輸出目錄，以免將既有 cache 當 cold。原始結果為本機忽略檔案，未必存在於其他 checkout：

- `artifacts/startup-profile/lazy-paired-live/runs.json`、`summary.json`
- `artifacts/startup-profile/lazy-paired-live-final/runs.json`、`summary.json`
- `artifacts/startup-profile/lazy-baseline-source/`

## 7. 尚待驗證與額外發現

- Full 專屬實作的真實執行驗證仍待對應環境，不在 light 結果中冒充完成。
- 進一步效能工作應測整段問答，並分開追蹤必要 Core／SDK 載入、中文資源與 Windows I/O；不以延後成本就認定消除成本。
- 盤點完整 feature 分支與純工具拆檔仍可續做；目前不為了追求秒數擴大成全案重寫。
- 測 BM25 時發現既有 `BaseRetriever.invoke()` 會因其 `metadata` 是 list 而失敗；Akasha 使用的 `get_relevant_documents_and_scores()` 路徑通過。此問題在修改檢索載入前已重現，不在本輪修改，應另行處理。
- 沒有提交 commit，沒有修改金鑰／`.env`、依賴 profiles 或使用者原有 `.gitignore` 變更。
