# akasha.agents Skills Phase 2 實作規劃（Skill Tool Bundle）

Date: 2026-07-24  
Status: 規劃稿，尚未實作

## 1. 目標

在 Phase 1 instruction-only skill 的基礎上，讓 Skill 可以提供額外的 tools，同時維持既有 tools 參數與行為：

    agent = akasha.agents(
        tools=[user_tool],
        skills=["research"],
    )

預期結果：

- user_tool 繼續可用。
- research skill 可以提供額外的已授權 tools。
- skill tools 只能從 registry allowlist 取得。
- skill 不會任意 import 或執行目錄中的 Python。
- tools 與 skill tools 發生名稱衝突時，不會靜默覆蓋。

## 2. 目前狀態

Phase 1 已經具備：

- Skill model。
- SkillRegistry。
- SKILL.md loader。
- SkillContext。
- skills 參數。
- instruction middleware。
- skill 名稱與版本 logs。

Skill model 目前已有 tools 欄位，但 Phase 1 尚未使用它。Phase 2 需要把這個欄位改成「經過 registry 驗證後的 tools」，不能直接信任外部 skill metadata。

## 3. 建議的 Skill Tool Bundle 模型

### 3.1 Skill 宣告

Skill 可以宣告 tool 名稱，而不是直接從 skill 目錄載入任意 Python：

    Skill(
        name="research",
        instructions="Use the research workflow.",
        tool_names=("web_search_tool", "rag_tool"),
    )

如果要維持目前的 Skill model，也可以暫時保留 tools 欄位，但 loader 只能產生 metadata，最後由 registry/resolver 注入已授權的 BaseTool。

建議長期模型：

    @dataclass(frozen=True)
    class Skill:
        name: str
        description: str = ""
        instructions: str = ""
        tool_names: tuple[str, ...] = ()
        tools: tuple[BaseTool, ...] = ()
        version: str = ""
        metadata: Mapping[str, Any] = field(default_factory=dict)

tool_names 是 skill 的宣告；tools 是 resolver 驗證後實際提供給 agent 的工具。

### 3.2 SKILL.md frontmatter

外部 SKILL.md 只使用標準 frontmatter：

    ---
    name: research
    description: Web and document research workflow
    ---

    Skill instructions...

外部 skill 不在檔案中宣告 version 或 tools。若應用程式需要 skill-specific tools，應透過 Python Skill 物件、agent 的既有 tools，或應用程式內部的受控綁定提供。
## 4. Tool Registry 與 Allowlist

新增或擴充 ToolRegistry，負責管理可被 skill 使用的工具：

    registry.register(
        "web_search_tool",
        lambda: websearch_tool(search_engine="brave"),
    )

    registry.register(
        "rag_tool",
        lambda: rag_tool(embeddings="openai:text-embedding-3-small"),
    )

建議 registry 儲存 factory，而不是共用已經執行過的 tool instance。

原因：

- 避免不同 agent 共用可變的 tool state。
- 可依 agent 的 env_file、language、embedding 設定建立 tool。
- 讓每次 agent 初始化都能取得明確且可測試的工具集合。

### 4.1 Allowlist 規則

- 只有 registry 中已註冊的名稱可被 skill 使用。
- skill 宣告未知名稱時，在 agent 初始化時直接報錯。
- tool factory 回傳值必須是 BaseTool。
- registry key 必須與 BaseTool.name 一致，或註冊時明確指定 alias。
- registry 不執行任意 import path。
- skill 不可以覆蓋或修改 registry 中其他 tool。

### 4.2 Factory context

部分 Akasha tools 需要 agent 設定，例如 env_file、language、embeddings。Phase 2 應支援受控的 factory context：

    ToolFactory = Callable[[SkillToolContext], BaseTool]

    @dataclass(frozen=True)
    class SkillToolContext:
        env_file: str = ""
        language: str = "ch"
        model: str = ""

不應把整個 agents 物件傳入 factory，避免 skill tool 依賴 agent 內部實作。

## 5. Resolver 設計

Skill resolver 的責任：

1. 解析 skills 參數。
2. 載入 skill metadata。
3. 將每個 tool name 交給 ToolRegistry。
4. 建立 skill tool instances。
5. 與呼叫端傳入的 tools 合併。
6. 檢查 tool name collision。
7. 產生最終的 SkillContext 與 tool list。

建議介面：

    resolve_skill_tools(
        skill_context,
        registry,
        tool_context,
        existing_tools,
    ) -> ResolvedSkillTools

建議結果模型：

    @dataclass(frozen=True)
    class ResolvedSkillTools:
        tools: tuple[BaseTool, ...]
        skill_tool_names: Mapping[str, tuple[str, ...]]

既有 tools 必須先建立並保留。skill tools 只能追加，不能覆蓋。

## 6. Tool 名稱衝突策略

預設採 fail fast：

    ValueError(
        "skill tool name 'search' conflicts with an existing agent tool"
    )

需要檢查：

- 呼叫端 tools 之間的重複名稱。
- 不同 skills 之間的重複名稱。
- skill tool 與既有 tools 的重複名稱。
- registry alias 與實際 BaseTool.name 不一致。

不建議自動加 prefix，例如 research_search，因為會改變模型看到的公開 tool schema；如果未來需要 namespace，應另行設計明確的 opt-in 機制。

## 7. LangChain 整合方式

### 7.1 優先方案：建立 agent 時合併 tools

若 skills 在 agent 初始化時已確定，優先將解析後的 tools 傳入：

    create_agent(
        model=self.model_obj,
        tools=existing_tools + skill_tools,
        middleware=[skill_prompt_middleware(...)],
    )

這是最簡單且最容易測試的路徑，並且能保留目前的 tool calling、stream 與 async 行為。

### 7.2 動態工具方案：middleware

只有在「每次 model call 可能有不同 skill tools」的需求成立時，才使用 LangChain dynamic tool middleware：

- model request 階段加入可見 tools。
- tool call 階段根據 tool name 找到並執行對應 tool。
- 未知 tool call 必須回傳可理解的 tool error，不可靜默忽略。
- middleware 必須支援同步與非同步執行路徑。

Phase 2 第一個實作版本建議採 7.1；dynamic tool middleware 保留為 Phase 2b。因為目前 skills 是 agents 初始化參數，不需要每次 call 改變 tool schema。

## 8. Agents API 變更

維持：

    akasha.agents(
        tools=existing_tools,
        skills=skills,
    )

新增內部流程：

    self.skill_context = resolve_skills(skills)
    self.skill_tools = resolve_skill_tools(
        self.skill_context,
        existing_tools=self.tools,
        tool_context=...,
    )
    self._agent = create_agent(
        model=self.model_obj,
        tools=self.skill_tools,
        middleware=[skill_prompt_middleware(...)],
    )

注意：

- self.tools 仍應代表呼叫端傳入的原有 tools，避免破壞既有 logs 與診斷語意。
- 可以新增 self.skill_tools，或另外記錄 skill tool mapping。
- _set_model 重新建立 agent 時，不應重複建立同一批有狀態的 tool；要先決定是否重新 factory。
- tools 參數仍然接受現有 BaseTool / StructuredTool / MCP tools。

## 9. Async 與 Streaming

需要分類：

### Sync-capable tool

同時提供 func 或可同步 invoke。既有 stream 流程可繼續使用。

### Async-only tool

只有 coroutine，例如部分 MCP tools。需要遵守目前 agents 的 stream guard：

- stream=False 使用 ainvoke()。
- stream=True 若同步 stream 不支援，初始化或第一次呼叫時明確拒絕。
- 錯誤訊息要說明哪個 skill tool 是 async-only。

### Tool execution errors

skill tool 執行失敗時：

- 保持 LangChain ToolMessage / error handling contract。
- logs 記錄 skill、tool name 與錯誤。
- 不把 secrets 或完整環境設定寫入 logs。
- 不在 resolver 階段執行 tool，只建立 tool instance。

## 10. Logs 與可觀測性

保留既有欄位並新增：

    {
        "skills": ["research"],
        "skill_versions": {
            "research": "1.0.0"
        },
        "skill_tools": {
            "research": ["web_search_tool", "rag_tool"]
        }
    }

建議不要只記錄最終 tools list，因為需要區分：

- 呼叫端明確提供的 tools。
- skill 注入的 tools。
- 兩者是否發生過衝突或拒絕。

## 11. 測試規劃

### Unit tests

新增或擴充 tests/unit/test_agent_skills.py：

- ToolRegistry register / lookup。
- factory 非 BaseTool 時拒絕。
- 未知 tool name 被拒絕。
- tool factory 使用受控 context。
- 同一 skill 的 tool 去重。
- 不同 skills 的 tool name collision。
- skill tool 與既有 tool collision。
- resolver 不會執行 tool，只建立 instance。
- SkillContext 正確列出 skill tool names。

### Contract tests

新增或擴充 tests/contract/test_agent_skills_contract.py：

- tools=[]、skills=[] 的既有行為不變。
- tools 與 skill tools 同時被傳給 create_agent。
- skill prompt 與 skill tools 同時存在。
- skill tool 可被 fake model 呼叫。
- non-stream tool calling 維持原有結果。
- stream tool event 維持原有格式。
- logs 可以 JSON serialize。
- skill_tools logs 不會洩漏 tool secrets。

### MCP / async tests

- async-only skill tool 在 non-stream agent 中可執行。
- sync stream 遇到 async-only skill tool 時明確拒絕。
- 多個 MCP skill tools 不會互相覆蓋。
- tool error 仍能轉成 ToolMessage。

## 12. 實作階段

### Phase 2a：Static skill tool bundles

1. 建立 ToolRegistry 與 ToolFactory 型別。
2. 擴充 Skill metadata，支援 tool_names。
3. 實作受控 ToolFactory context。
4. 實作 resolver 合併既有 tools 與 skill tools。
5. 在 agent 建立時傳入完整 tool list。
6. 實作 collision、allowlist、factory error。
7. 增加 logs 與 unit/contract tests。

### Phase 2b：Dynamic tool middleware（視需求）

1. 確認是否真的需要每次 call 變更 skill tools。
2. 實作 model request 的 dynamic tools。
3. 實作 tool call routing。
4. 補足 sync / async 與 error handling。
5. 加入 middleware ordering 與 regression tests。

## 13. 不在 Phase 2 處理

- 不讓模型任意呼叫 load_skill。
- 不執行 skill 目錄中的 Python。
- 不接受任意 Python import path 作為 tool factory。
- 不自動下載遠端工具或套件。
- 不靜默覆蓋既有 tools。
- 不改變原有 tools 參數的公開語意。
- 不在沒有實際需求前加入每次 query 的 dynamic tool schema。
## 14. 待確認決策

已確認的決策：

- 外部 skill、SKILL.md、SKILL.md frontmatter 只使用 tool_names metadata，不保存 Python tool instances。
- 程式直接建立的 Skill 可以接受 tools instances，作為受信任的 Python API escape hatch。
- resolver 最終一律產生獨立的 resolved tool instances，再交給 create_agent。
- Skill 負責描述需要什麼；ToolRegistry 負責決定允許提供什麼；resolver 負責建立實際工具。

仍待確認：

1. 是否採用 Phase 2a 的 static tool merge 作為第一版？
2. ToolRegistry 是否由呼叫端顯式建立，或先使用全域 default registry？
3. tool factory 是否需要 SkillToolContext 的 env_file、language、model？
4. tool name collision 是否維持 fail fast？
5. 是否有實際需求在同一個 agent instance 中，依不同 query 動態改變 skill tools？

### 14.1 決策後的模型

    Skill definition
      tool_names = ["web_search_tool", "rag_tool"]
              |
              v
    Skill resolver + ToolRegistry
              |
              v
    ResolvedSkill
      tools = (actual BaseTool instances)
              |
              v
    create_agent(...)

建議的資料模型：

    @dataclass(frozen=True)
    class Skill:
        name: str
        instructions: str = ""
        tool_names: tuple[str, ...] = ()
        tools: tuple[BaseTool, ...] = ()  # trusted Python API only

tools 欄位不由檔案 loader 填入，也不應繞過 registry；它只支援測試或呼叫端已明確建立的受信任 Skill。所有 tools 都必須在 resolver 階段通過名稱衝突檢查，並轉成 agent 專用的 resolved tool collection。
## 15. 已確認的產品方向：先以單一 Agent 為主

目前大部分使用情境是由單一 agent 完成任務，因此 Phase 2 不預留或公開完整的 multi-agent orchestration API。

### 15.1 公開使用方式

使用者只需要指定 skill 名稱：

    agent = akasha.agents(
        tools=[existing_tool],
        skills=["research"],
    )

不公開要求使用者建立或傳入 SkillRegistry。Skill 的 discovery、載入、metadata 驗證與 tool allowlist 由 Akasha 內部處理。

### 15.2 內部責任邊界

    skills=["research"]
            |
            v
    Akasha internal SkillCatalog / Registry
            |
            v
    immutable Skill definition
            |
            v
    per-agent SkillContext
            |
            v
    per-agent resolved tool instances
            |
            v
    create_agent(...)

規則如下：

- Skill definition 是可重用的 immutable 描述資料。
- 每個 agent 解析自己的 SkillContext。
- 每個 agent 建立自己的 resolved tool instances，不共用可能帶狀態的 tool instance。
- registry/catalog 是內部實作，不是使用者必須理解的 API。
- 測試或進階 Python 呼叫仍可直接傳入 Skill 物件。

### 15.3 Phase 2 不處理 multi-agent

Phase 2 不加入以下項目：

- supervisor agent。
- agent-to-agent handoff。
- agent routing 或 agent ownership。
- 跨 agent shared memory。
- 每個 agent 的公開 registry 管理 API。

未來若需要 multi-agent，會在更高層增加 orchestration layer，例如由 router 將工作分派給多個既有的 akasha.agents instances；現有 skills API 不需要因此改變。

### 15.4 對原待確認項目的影響

- 不再把 skill_registry 作為公開的 agents 參數。
- default registry/catalog 由 Akasha 內部管理。
- static skill tool merge 仍是 Phase 2a 的優先方案。
- dynamic tool middleware 與 multi-agent orchestration 都延後到有明確 use case 時再規劃。
## 16. 路徑型 Skill 載入決策

skills 參數支援三種輸入：

- Skill 物件：直接使用受信任的 Python Skill。
- 存在的目錄路徑：呼叫 load_skill_directory，載入本次 agent 專屬的 SkillContext。
- 其他字串：視為 skill name，從 Akasha 內部 catalog 解析。

推薦使用方式：

    agent = akasha.agents(
        skills=["skills/research"],
    )

路徑型載入不會呼叫 default_registry.register()。載入結果只屬於當次 agent，避免全域 registry 污染、重複註冊與不同 agent 互相影響。外部 SKILL.md 不宣告 tools；實際工具由 agent tools 或應用程式內部受控綁定提供。

如果字串看起來是路徑但目錄不存在，resolver 應直接回報 FileNotFoundError，不應靜默把它當成 skill name。
## 17. Standard Skill Resources

外部 filesystem skill 遵循 Agent Skills 標準，只從 SKILL.md frontmatter 取得 metadata 與 instructions。references/、assets/、scripts/ 不會在初始化時整包載入。

skill 載入後，DynamicSkillMiddleware 才提供 read_skill_resource(skill, path)：

- path 只能是 skill root 下的相對檔案路徑。
- 只讀取 UTF-8 文字檔。
- 不執行 scripts。
- binary resource、directory、path traversal 與超過大小上限的檔案會被拒絕。
- 既有 tools 與 Python Skill 的 tool bundle 行為維持不變。
