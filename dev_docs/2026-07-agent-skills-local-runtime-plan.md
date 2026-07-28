# Agent Skills 本機 Runtime 與 Script 執行修改計畫

## 目標

讓 Akasha 在保留現有 `tools` 與 `skills` 公開介面的前提下，完整支援受信任本機 skill 的 script 執行。

使用者只需要指定 skill 路徑：

```python
import akasha

agent = akasha.agents(
    tools=[search_tool],
    skills=["skills/research"],
)
```

Skill script 預設使用「呼叫端主程式目前正在執行的環境」，而不是 Akasha 自己的獨立 runtime。只要呼叫端環境已安裝 skill 所需套件，script 就可以使用；Akasha 不負責替使用者建立環境或自動安裝依賴。

## 設計決定

### 公開 API

- 保留既有 `tools=[...]`，行為不變。
- 保留既有 `skills=[...]`，不新增必要的 `skill_executor`、`sandbox` 或 runtime 參數。
- skill 目錄本身代表呼叫端已授權 Akasha 使用該本機 skill。
- script 執行能力由 Akasha 內部 middleware 動態提供，不需要使用者自行註冊 execution tool。

### 主程式執行環境

Skill script 應使用呼叫端主程式的 runtime：

- Python script 使用呼叫端的 `sys.executable`。
- 子程序繼承呼叫端的 environment variables 與可用的 PATH。
- script 可以使用呼叫端環境中已安裝、但 Akasha 本身沒有依賴的套件。
- Akasha 不自動執行 `pip install`、修改環境或建立虛擬環境。
- 缺少套件、interpreter、OS capability 或外部 binary 時，回傳明確的執行錯誤給 agent。

這裡的 runtime 是呼叫端應用程式的 runtime，不稱為「Akasha runtime」。

### Agent 可見工具

初始 turn 維持目前的 progressive disclosure：

1. 只有 `load_skill` 與呼叫端提供的既有 tools。
2. `load_skill` 成功後，載入完整 `SKILL.md` instructions。
3. Skill 載入後，動態提供 resource reader 與 script execution capability。

Script 不會自動各自註冊成 Python tool。模型透過一個由 Akasha 內部提供的受限制 execution tool，依照 `SKILL.md` 指定的相對路徑與參數執行 script。

## 實作步驟

### Phase A：內部執行介面

1. 在 skills middleware 內新增內部 script execution handler。
2. 明確定義 execution request 與 result：
   - skill name
   - skill root 下的相對 script path
   - 結構化 argument list
   - exit code
   - stdout
   - stderr
3. 只允許已載入的 filesystem skill。
4. 使用安全的相對路徑解析，拒絕絕對路徑、`..` traversal、directory path 與 symlink escape。
5. Python script 使用主程式的 `sys.executable`；其他 script 依照明確 interpreter 或 PATH 執行。

### Phase B：執行限制與錯誤處理

1. 不接受任意 shell command 字串，避免 execution tool 變成未限制的 shell。
2. 新增 timeout、stdout/stderr 上限與 process return code 處理。
3. 執行失敗時回傳可供 agent 理解的錯誤，包括：
   - script 不存在；
   - interpreter 不存在；
   - Python module 不存在；
   - process timeout；
   - non-zero exit code。
4. 不自動安裝依賴、不自動下載程式、不修改呼叫端環境。
5. binary output 不直接當成 UTF-8 文字處理；必要時回傳受控的錯誤或摘要。

### Phase C：Skill instructions 與 resource 整合

1. 更新使用文件，說明 `SKILL.md` 必須描述：
   - 何時執行 script；
   - script 的相對路徑；
   - 參數格式；
   - 預期輸出與錯誤處理；
   - 所需 Python 套件、interpreter 或外部 binary。
2. 保留 `references/`、`assets/` 與 `scripts/` 的按需載入。
3. `read_skill_resource` 負責讀取文字 resource；execution tool 負責執行 script，兩者職責分離。
4. 不新增目錄自動索引，也不讓 loader 預先載入所有 script。

### Phase D：測試與相容性

新增或更新測試：

- Python script 使用呼叫端 `sys.executable` 執行。
- script 可以 import 呼叫端已安裝、但 Akasha 未直接依賴的套件。
- script 收到結構化參數並正確回傳 stdout/stderr/exit code。
- 初始 turn 不包含 script execution tool。
- skill 載入後才包含 script execution capability。
- 拒絕 skill root 外的 script、絕對路徑、`..` 與 symlink escape。
- timeout 與輸出大小限制有效。
- 缺少 interpreter 或 module 時回傳明確錯誤，不讓 agent process 崩潰。
- 既有 `tools`、Python `Skill` tools 與 resource reader 行為不變。
- 不自動安裝依賴或執行 `skill.yaml`。

## 不在本次範圍

- 不新增公開的 `skill_executor` 參數。
- 不引入 sandbox、container 或遠端 execution service。
- 不處理 multi-agent 或 agent-to-agent 的 runtime 隔離。
- 不根據 `allowed-tools` 自動建立工具。
- 不自動掃描或執行 skill 目錄中的所有程式。
- 不自動安裝 skill dependencies。

## 未來擴充方向

目前先把 execution 實作封裝在 Akasha 內部，保留未來替換執行後端的可能性。若將來需要支援不受信任 skill、multi-agent、agent-to-agent 或 production isolation，再評估接入 LangChain/Deep Agents 的 backend 或 sandbox；屆時不應要求修改現有 `skills=[...]` 使用方式。
