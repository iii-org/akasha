# Akasha 全功能延遲載入與模組責任整理 PRD

- 日期：2026-10-09
- 狀態：第一輪載入整理已實作，light 回歸與 Gemini 真實測試完成；full runtime 待驗證，穩定端到端加速尚未證明。
- 修改範圍：全版 Akasha，共用程式碼與選配功能均納入，不限於 light。
- 目前測試環境：先使用基礎／light 環境；full 專屬功能的執行驗證另列待辦。
- 相容性原則：保留既有使用方式與 LangChain 生態系，不改成直接 Google SDK 的替代架構。

## 1. 背景與問題

使用者觀察到 Akasha 首次問答可能耗時約 30 秒，而直接使用 Google SDK 時沒有相同體感。調查發現，首次載入的成本包含套件匯入、Python bytecode 編譯與快取檔案 I/O、模組初始化，以及實際模型請求；不能將整段時間都解讀為模型回應時間。

Akasha 的頂層匯出與部分 provider 已採延遲載入，但共用 helper、基底類別、文件入口及 Agent／Skills 套件入口仍有連帶匯入。例如，簡單工具名稱解析會載入中文處理資源，取得 `create_tool` 會載入 Agent／Skills 與 REPL 相關實作。

單純將 import 分散搬進各函式，雖可減少部分首次載入，卻可能使依賴位置、初始化時機與錯誤行為難以理解。因此本次應同時整理功能責任，建立一致且可測試的載入規則。

## 2. 目標與非目標

### 2.1 目標

1. 使用某項功能時，才載入其所需的實作與第三方套件，避免載入不相關功能。
2. 讓維護者能從功能模組與選擇分支直接看出依賴，不需理解自製動態載入框架。
3. 延伸現有 provider 延遲載入模式，涵蓋全版功能，而非維護 light／full 兩份實作。
4. 保留公開 import 路徑、呼叫方式、回傳格式、串流行為與既有 Agent 流程。
5. 以功能相容性、載入範圍與可重現的效能量測共同驗收。

### 2.2 非目標

- 不替換 LangChain，不自行重建 Agent 執行引擎。
- 不移除 LangSmith 或 monkeypatch 第三方依賴以跳過正常匯入。
- 不在本次重新定義 light／full 安裝 profiles、Python／NumPy 支援範圍或移除必要套件。
- 不修改使用者的防毒、Windows 檔案系統設定或停用 Python bytecode 快取。
- 不保證所有首次操作低於固定秒數，也不保證實際需要的依賴成本消失。
- 不順帶重寫模型、搜尋、評分或 Agent 演算法。

## 3. 已有證據與限制

本節記錄調查背景，不作為跨機器效能承諾。

| 證據 | 可支持的結論 | 不能推論的事項 |
| --- | --- | --- |
| 單次完整首次就緒 23.37 秒，其中快取寫入相關約 16.79 秒 | 快取 I/O 可占很大比例 | 所有冷啟動都固定耗時 23.37 秒 |
| 同次 LangSmith 匯入分支約 3.79 秒 | LangSmith 路徑是部分成本 | 它是全部延遲來源，或整段分支都能移除 |
| Windows ETW 對應到部分慢 open／write 請求 | 部分等待確實跨越 Windows File I/O | 已證明是 ESET、實體磁碟或特定 minifilter 阻塞 |
| 獨立 Core、直接 Google SDK 與 Akasha 都出現冷快取成本 | 問題不只存在於 Akasha | 三者工作負載完全相同 |
| Akasha 匯入／建構路徑有更多快取寫入與功能連帶載入 | 減少不必要依賴有改善空間 | 套件數與延遲是固定線性關係 |

在同一現有環境、三次輪替順序、各自空 bytecode 快取的診斷中，首次就緒中位數為 Core 約 12.56 秒、直接 Google SDK 約 10.42 秒、Akasha 約 21.92 秒。Core 使用本地 fake model；直接 SDK 與 Akasha 只建立 client，沒有雲端模型請求。這不是功能等價的問答比較，也不是全新機器或 OS 快取清空測試。

目前 runtime 測試使用基礎／light 環境，不含 Torch、Transformers、MLflow、Streamlit、FAISS 等已檢查的選配套件。原始碼盤點已跨問答、RAG、摘要、搜尋、Agent／Skills、評分、圖片、Memory、API 與部分選配實作，但不代表全版每項功能都已執行。

目前 `pyproject.toml` 的 `light = []`；基礎與 light 使用同一份程式碼及基礎 dependencies。安裝套件與執行時載入是不同議題。本 PRD 以目前檔案為準，不沿用舊文件對預設 full 安裝的描述。

## 4. 架構與共同寫法

### 4.1 三種載入模式

| 情境 | 統一寫法 | 注意事項 |
| --- | --- | --- |
| 公開套件入口 | 沿用明確匯出表與 `__getattr__` | 保留 `__all__`、型別資訊與既有物件身分；父套件入口也須輕量 |
| 選用功能／provider／策略 | 在實際選定的函式或分支使用明確 import | 不因載入 dispatcher 而載入所有實作 |
| 昂貴且可重用的資源 | 小型私有 getter，首次建立後重用 | 明確處理併發、失敗重試與生命週期；只有安全的資源才共用 |

實作例外：Agent facade 類別與 `agents.py` 子模組同名，全部套用延遲匯出會改變部分匯入順序下的類別身分。本輪保留輕量 facade 類別的固定匯出，改在建構時載入 LangChain graph runtime；其餘 Agent／Skills 匯出仍按需解析。驗收應檢查未載入 graph／middleware／REPL，而非要求 facade 類別定義完全不出現。

函式內 import 不需另加模組快取，Python 已會重用匯入的模組。只有昂貴物件初始化需要考慮額外快取。不得將帶有使用者設定、可變狀態或非執行緒安全的 client 任意做成全域 singleton。

不建立通用 plugin loader、透明代理物件或全面以字串動態解析函式。只有真正重複且適合集中管理的規則才抽出小型共用 helper；不以統一格式為理由增加抽象層。

### 4.2 模組責任

- 純工具：名稱解析、字串與輕量資料轉換，不初始化中文語料、網路、資料庫或模型資源。
- 功能入口：驗證選項並選擇實作，不預先載入所有 provider／格式／策略。
- 功能實作：持有實際需要的第三方依賴與資源建立流程。
- 公開匯出：維持原使用方式，內部模組整理不外洩成新操作步驟。

模組拆分以真實功能責任為單位，不為每個 import 新增檔案。現有共用基底可繼續存在，但不應因繼承就預先載入 DB、crawler 或 REPL。

### 4.3 相容性與錯誤

- 保留 `akasha.ask`、RAG、Agent、Skills、Memory 與 helper 的既有公開使用方式。
- 型別專用匯入可使用 `TYPE_CHECKING`／延後 annotation，但 runtime `isinstance` 所需型別不得誤移除。
- 缺少選配依賴時，沿用現有 `OptionalDependencyError` 與安裝提示規則，不以廣泛 `except ImportError` 隱藏實作錯誤。
- 不相關功能不應因缺少選配依賴而失敗；選定該功能後應在既有驗證位置或有紀錄的明確位置失敗，避免先完成昂貴工作才報錯。
- 既有套件入口、內部跨模組引用及 monkeypatch 測試需盤點；不能僅保留頂層名稱而忽略原引用行為。
- 區分可延後的匯入與不可任意延後的設定驗證、註冊副作用及資源釋放。

## 5. 全版修改範圍

| 領域 | 需求 | 保留事項 |
| --- | --- | --- |
| `atman.py` 與 DB | DB 實作在實際 DB 路徑載入 | RAG、Memory 與繼承功能的 DB 行為 |
| helper | 分離純工具與中文資源；Jieba／OpenCC 按需建立 | 中文長度、簡繁轉換與既有引用相容性 |
| 文件入口 | 純文字不載入 crawler；URL／格式選定後載入處理器 | 字串路徑判定、Document、混合 info、metadata 與目錄錯誤規則 |
| Agent／Skills | 輕量公開匯出；不使用 Skills 時不載入其 middleware | LangChain `create_agent` 流程、工具與 registry 生命週期 |
| Skills REPL | 執行 REPL 時才載入 experimental REPL 實作 | 原執行、重用與錯誤行為 |
| 檢索 | 依選定策略載入；auto／merge 載入其實際組合 | 選項驗證、排序與結果語義 |
| 評分 | 依選定 metric 載入 ROUGE／BERTScore 等依賴 | 語言處理、分數與缺依賴提示 |
| 圖片 | 預覽與 provider 處理各自載入所需實作 | 生成／編輯、圖片儲存與預覽行為 |
| API | 避免純雲端請求為 cleanup 首次載入 Torch | 真正使用本地 GPU 後仍執行必要清理 |
| 選配功能 | 檢查 UI、MLflow、FAISS、本地模型／embedding／rerank 的載入路徑 | 已有正確延遲載入者不為統一形式而重寫 |

以上是全版範圍，不以 light 是否安裝某套件決定是否整理其程式碼。完整功能盤點仍須在各批實作時補齊；本表不是所有分支已驗證的聲明。

## 6. 分批交付

### 第一批：共用核心與文件載入

- 整理 `atman.py`、helper 純工具／中文資源、文件 dispatcher 與 crawler。
- 同時驗證 ask、summary、RAG 與 Agent 的共用路徑，不將 `atman.py` 單獨視為完整解法。
- 交付責任分配說明、相容性測試、載入範圍測試與前後效能結果。

### 第二批：Agent 與 Skills

- 整理 Agent／Skills 套件匯出、middleware 與 REPL 載入位置。
- 驗證 `create_tool`、SkillRegistry、無 Skills Agent、啟用 Skills Agent 及 REPL 路徑。
- 保留 default registry、物件身分與既有 Agent 執行流程。

### 第三批：策略、評分與選配實作

- 整理 retrieval、metrics、圖片預覽與 API cleanup；盤點全版其餘選配入口。
- 對本來已按需載入的實作補充測試，不做無收益的重寫。
- 每批獨立 review 與回歸；不得混入無關演算法或依賴版本升級。

## 7. 驗收與測試策略

### 7.1 載入範圍

使用獨立 subprocess，避免測試順序與 `sys.modules` 污染。依 feature 的合法最小依賴定義載入範圍，而非要求所有共同依賴都不能出現。

至少涵蓋：

- `import akasha` 不載入主要功能實作與選配堆疊。
- 純名稱工具不載入 Jieba／OpenCC 資源實作。
- 純文字文件處理不載入 crawler／向量 DB 實作；URL 處理正常載入 crawler。
- `create_tool`、SkillRegistry 不順帶載入 Agent graph runtime／Skills middleware／REPL；允許輕量 facade 定義以保留匯入相容性。
- 無 Skills 的 Agent 不載入 Skills middleware／REPL；啟用後功能正常。
- 選定 provider、retriever、metric 不載入不相關實作。
- 未使用本地模型的 API cleanup 不因清理而首次載入 Torch。

斷言應聚焦 Akasha 可以控制的直接／連帶載入，不對第三方內部模組布局建立過度脆弱的契約。

### 7.2 功能相容性

- 覆蓋公開 import、建構、實際使用三個階段，不能只有 import 成功。
- 覆蓋 info 的空值、純文字、URL、路徑、Document、混合列表與缺少 handler。
- 覆蓋中文／非中文、串流／非串流、檢索組合、評分選項與既有錯誤行為。
- 驗證昂貴共享資源的併發建立與初始化失敗後行為。
- 以 fake／stub 驗證流程並明確標示，不能把它當成真實雲端或 full 執行證據。

### 7.3 目前先以 light 測試

先執行 light 可運作的功能回歸、載入範圍與缺少選配依賴測試。對 full 專屬功能，先使用必要的隔離測試／stub 檢查 dispatch、延遲時機及錯誤契約。

未安裝依賴而 skip 的案例須列為「未驗證」，不能列入通過數冒充 full 相容性。涉及本地模型、GPU、MLflow、UI、FAISS 等實際執行的結果，待對應環境補驗。每批報告應分別標示「程式碼修改完成」、「light 驗證完成」與「full runtime 待驗證」。

### 7.4 效能量測

- 量測前後使用相同 Python、依賴版本、案例、初始化參數與追蹤設定。
- 分別記錄 package import、公開功能解析、建構、第一次實際使用及後續使用，避免只是把成本移到下一階段。
- 區分新程序＋既有 bytecode、新程序＋空 bytecode、同程序重用；不要稱空 bytecode 測試為完整全新環境。
- 冷快取使用獨立目錄，至少三次循序、輪替前後順序；報告中位數、範圍、匯入模組與寫入數。不得並行執行以免 I/O 互相干擾。
- 診斷案例可不發網路請求；真正問答須另測並區分網路／模型延遲，不混用為同一指標。
- 至少以未開啟 WPR／ETW 的結果評估改善；系統追蹤只作深入診斷。
- 驗收要求不相關功能載入減少且無行為回歸；若時間變化落在波動內，應寫「未證明加速」，不誇大收益。

## 8. 風險與維護要求

1. 首次使用必要依賴仍有成本：報告整個操作，不僅比較建構時間。
2. 延後錯誤會影響體驗：保留明確的功能選擇與依賴驗證位置。
3. 套件父入口可能重新引入重依賴：檢查完整匯入路徑，不只修改單檔。
4. 全域資源可能有競態與狀態污染：getter 快取需依資源安全性設計，不全面套用 singleton。
5. 原引用位置變動可能破壞測試／下游：保留必要的相容匯出，並更新有實作細節耦合的測試。
6. Light 測試不能證明 full：保留待驗證清單，不因本機無套件而跳過全版設計。

每批應提供簡短的模組責任／依賴方向說明與功能測試。只有真正改變使用者操作或安裝方式時才修改使用者文件。不得修改無關 worktree 內容或把診斷 artifact 當正式套件功能。

## 9. 相關紀錄與證據位置

- [既有 provider 延遲載入紀錄](2026-08-09-provider-lazy-import-optimization.md)
- [本輪實作與測試結果](2026-10-09-feature-lazy-loading-worklog.md)
- [舊 light 升級計畫](upgrade_light_version.md)：僅作歷史背景；目前 packaging 以 `pyproject.toml` 為準。
- 本機診斷報告：`artifacts/startup-profile/DEPENDENCIES.md`、`LANGSMITH.md`、`WINDOWS_IO.md`、`CORE_ONLY.md`。
- 本機診斷腳本：`scripts/profile_startup.py`、`profile_langsmith_cost.py`、`compare_core_startup.py` 與 Windows I/O tracing scripts。

診斷 artifacts 為本機忽略檔案，未必存在於其他 checkout；本 PRD 已摘錄關鍵證據與限制。系統 ETW 可能包含檔案路徑等資訊，不直接公開上傳。

## 10. 本次紀錄的決策

- 採小範圍、分批的模組責任整理搭配延遲載入，不採全案重寫或零散緊急補丁。
- 修改範圍涵蓋全版，初期測試先用 light；兩者不可混為同一範圍。
- 公開使用方式與 LangChain 架構保持不變。
- 初始 PRD 僅記錄設計；後續實作與實測另見 worklog。不得將完成載入整理等同於已解決約 30 秒的冷啟動問題。
