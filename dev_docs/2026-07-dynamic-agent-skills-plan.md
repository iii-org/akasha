# akasha.agents 動態 Skills 實作計畫（Draft）

Date: 2026-07-24  
Status: 待確認，尚未進入實作

## 1. 目標

保留既有 tools 參數與行為，新增可選的 skills 參數：

    agent = akasha.agents(
        model="openai:gpt-5.4",
        tools=[search_tool, rag_tool],
        skills=["research", "reporting"],
    )

Skill 提供任務知識、工作流程與使用規則；Tool 仍是實際執行外部動作的 LangChain BaseTool。Skill 不取代 Tool，也不改變既有 tools API。

## 2. 第一版範圍

第一版先支援：

1. Instruction skill：由 SKILL.md 提供 instructions，在模型呼叫時加入 system prompt。
2. Tool-bundle skill：skill 可宣告已註冊的工具，但工具仍須是 BaseTool，並經過 allowlist 授權。

第一版不允許從任意路徑直接執行 Python，也不讓模型自行載入未授權套件。

## 3. 公開 API

在 akasha.agent.agents.agents.__init__ 增加：

    skills: Union[str, Skill, Sequence[Union[str, Skill]], None] = None

建議接受：

- None：不啟用 skill，現有行為不變。
- "research"：依名稱從 registry 載入。
- Skill：呼叫端已載入的 skill。
- list / tuple：啟用多個 skill。

Path 是否開放，列為待確認的安全決策。

本次可順便將 tools 的 mutable default 從 [] 改為 None，保持傳入 list 的相容性。

## 4. 建議資料模型

    @dataclass(frozen=True)
    class Skill:
        name: str
        description: str
        instructions: str
        tools: tuple[BaseTool, ...] = ()
        version: str = ""
        metadata: Mapping[str, Any] = field(default_factory=dict)

Skill 是 Akasha 自己的資料模型，不直接綁定 LangChain agent runtime；只有 tools 欄位使用 BaseTool。

## 5. 建議模組

新增 akasha/agent/skills/：

    __init__.py       # Skill 與公開 API
    models.py         # Skill / SkillMetadata
    loader.py         # skill 載入
    registry.py       # 名稱、版本、來源、allowlist
    resolver.py       # 去重、衝突檢查、合併
    middleware.py     # LangChain prompt / tool 整合

agents.py 只負責接收 skills、建立 skill context、傳入 middleware，以及記錄啟用的 skills。

## 6. 執行流程

初始化時：

    agents(tools=..., skills=...)
        -> normalize skills
        -> registry / loader 載入與驗證
        -> resolver 產生 SkillContext
        -> create_agent(model, tools=既有 tools, middleware=...)

既有 tools 永遠保留。Skill 的額外工具只能加入，名稱衝突預設 fail fast，不覆蓋呼叫端傳入的 tool。

每次呼叫時：

    agent(question)
        -> middleware 取得 SkillContext
        -> 將 skill instructions 加入 system prompt
        -> model 使用既有 tools / skill tools
        -> 保持目前 response、stream、thinking、logs 流程

第一版不做依 query 自動選 skill；未來可在 resolver 增加 resolve(question, context)。

## 7. LangChain 整合

Akasha 目前使用 langchain.agents.create_agent。LangChain 1.x middleware 支援 runtime system prompt 與 dynamic tools，因此適合此設計。

建議拆成兩個 middleware：

    SkillPromptMiddleware(skill_context)
    SkillToolMiddleware(skill_context)

Phase 1 只實作 SkillPromptMiddleware。Phase 2 再加入 dynamic tool registration、tool execution、async-only tool 與 stream guard。

agents._build_agent() 保持單一建立點；模型切換重新建立 agent 時，skill context 必須沿用。

## 8. Skill 檔案格式

外部 skill 使用標準 SKILL.md frontmatter，不增加 Akasha-specific 欄位：

    skills/
      research/
        SKILL.md
      reporting/
        SKILL.md

SKILL.md：

    ---
    name: research
    description: Web and document research workflow
    ---

    Skill instructions...

外部 skill 只提供 metadata 與 instructions。Skill-specific tools 由 agent 的既有 tools、Python Skill 物件或應用程式內部受控綁定提供。
## 9. 相容性要求

- agents(tools=[...]) 不傳 skills 時行為完全不變。
- BaseTool、StructuredTool、MCP tools 仍可傳入 tools。
- sync / async invoke 行為不變。
- stream 的 answer、thinking、tool event 格式不變。
- logs 仍可 JSON serialize。
- 保留 tools log 欄位，新增 skills、skill_versions、skill_sources。
- tool 名稱衝突不可靜默覆蓋。
- skill 載入錯誤在初始化時明確報錯。

## 10. 測試計畫

新增 tests/unit/test_agent_skills.py：

- skills normalization。
- SKILL.md 解析與預設 metadata。
- 多 skill 合併、去重與衝突。
- 未知 skill、格式錯誤、allowlist 驗證。
- instructions 是否進入 model request。
- 沒有 skills 時沒有 middleware side effect。

在 contract tests 增加：

- tools 與 skills 可以同時使用。
- 既有只有 tools 的案例仍通過。
- skill tool 能產生正常 ToolMessage。
- non-stream、stream、thinking、logs contract 不變。

安全 regression：

- path traversal 被拒絕。
- 未註冊 tool 不因 metadata 宣告而載入。
- skill instructions 不會被當成 Python 執行。
- skills 與 tools 同名時不覆蓋既有 tool。

## 11. 實作階段

### Phase 1：Instruction-only skill

- Skill model、loader、registry、resolver。
- agents 增加 skills 參數。
- middleware 注入 instructions。
- logs、unit tests、contract tests。

### Phase 2：Skill tool bundle

- registry allowlist 與 tool factory。
- skill 額外 BaseTool。
- collision、async-only tool、stream guard。
- tool calling integration tests。

### Phase 3：Optional dynamic resolution

- 依 question / user context 選 skill。
- cache、版本、prompt token budget。
- 評估是否允許 agent 主動要求載入 skill。

## 12. 暫不處理

- 不移除或改名 tools。
- 不把所有 tools 自動轉成 skills。
- 不執行任意 skill 目錄中的 Python。
- 不做遠端 skill marketplace。
- 不自動授予 filesystem、network、secrets 權限。
- 不把 skill selection 與 LLM 自主決策綁死。

## 13. 待確認決策

1. 是否同意 Phase 1 先做 instruction-only skill？
2. skills 是否暫時只接受名稱與 Skill，不接受任意 filesystem path？
3. registry 預設來源是否限定為 akasha/skills/，或由呼叫端顯式傳入？
4. instructions 要每次 call 重讀，還是初始化時載入並 cache？
5. tool name collision 是否採 fail fast？本計畫預設採 fail fast。
## 17. Resource Loading

Skill directory 可包含 references/、assets/ 與 scripts/。Akasha 不會預先索引或執行這些內容；skill 載入後才提供 read_skill_resource，讓模型依 SKILL.md 中的相對路徑讀取 UTF-8 text resource。resource path 必須留在 skill root 內，並受 max_resource_bytes 限制。
