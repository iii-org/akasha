# akasha.agents 動態 Skills 使用說明

日期：2026-07-24

## 基本用法

akasha.agents 保留原有的 tools 參數，並新增 skills 參數：

    import akasha

    agent = akasha.agents(
        tools=[existing_tool],
        skills=["skills/research"],
    )

tools 與 skills 是兩個獨立的參數：

- tools：agent 建立時立即可用。
- skills：先提供 skill metadata，模型需要時再動態載入完整內容。

使用 skill path 時，不需要手動呼叫 load_skill_directory，也不需要自行管理 registry。

## Skill Directory 結構

最小的 skill directory 如下：

    skills/
    └── research/
        └── SKILL.md

## SKILL.md 格式

SKILL.md 必須以 YAML frontmatter 開頭。為了保持與其他 skill 生態系相容，frontmatter 只使用標準的 name 與 description：

    ---
    name: research
    description: Research workflow
    ---

    # Research

    Search for reliable sources.
    Separate verified facts from assumptions.
    Mention limitations when evidence is incomplete.

SKILL.md 是 instruction-only 格式，不宣告 version 或 tools。skill 需要的工具由 agent 的 tools 參數提供，或由 Python Skill 物件與應用程式程式碼綁定。
## Phase 2b：動態載入流程

當建立 agent：

    import akasha

    agent = akasha.agents(
        tools=[existing_tool],
        skills=["skills/research"],
    )

初始階段只會載入 SKILL.md 的 frontmatter。模型第一輪可以看到：

- 使用者傳入的 existing_tool
- load_skill tool
- skill 的名稱、描述與 tool names metadata

模型需要先呼叫：

    load_skill(reference="skills/research")

load_skill 完成後會更新 agent state。下一個 model turn 才會載入：

1. SKILL.md frontmatter 以下的完整 instructions。
2. agent 原本提供的 tools，以及應用程式另外綁定的工具。
因此，skill instructions 不會在 agent 建立時提前加入模型。filesystem skill 不會從 SKILL.md 自動宣告或建立工具。

## 最小完整範例

假設目錄如下：

    skills/
    └── research/
        └── SKILL.md

Python 程式：

    import akasha

    agent = akasha.agents(
        skills=["skills/research"],
    )

    response = agent("請研究這個主題，並清楚區分已驗證的事實與推測。")
    print(response)

模型的預期流程：

    model
      |
      | 呼叫 load_skill("skills/research")
      v
    更新 loaded_skills state
      |
      | 下一個 model turn
      v
    載入 instructions，並使用 agent 已提供的 tools
      |
      v
    呼叫 skill tool 並完成回答

## 使用內部 Skill 名稱

如果 skill 已經存在於 Akasha 的內部 skill catalog，可以傳入 skill name：

    import akasha

    agent = akasha.agents(
        skills=["research"],
    )

目前一般使用情境不需要自行建立或傳入 skill_registry。使用 filesystem path 時，Akasha 會直接從該 path 載入 frontmatter，並在模型呼叫 load_skill 後載入完整 instructions。

## 使用 Python Skill 物件

需要在程式中直接建立 skill 時，可以傳入 Skill：

    import akasha
    from akasha import Skill

    concise = Skill(
        name="concise",
        description="Answer concisely",
        instructions="Keep the answer brief and direct.",
    )

    agent = akasha.agents(
        skills=[concise],
    )

直接傳入的 Skill 也遵循 Phase 2b 流程：建立 agent 時先使用 metadata，模型呼叫 load_skill 後才會取得 instructions。Python Skill 仍可由程式碼設定 tool_names 或 tools。

## Skill Tool Factory

Skill tool 可以透過 ToolRegistry 使用 factory 建立：

    from akasha.agent.skills import SkillToolContext, ToolRegistry

    registry = ToolRegistry()

    def create_research_tool(context: SkillToolContext):
        return web_search_tool(
            language=context.language,
            env_file=context.env_file,
        )

    registry.register("web_search_tool", create_research_tool)

factory 可以取得目前 agent 的執行環境，例如：

- context.env_file
- context.language
- context.model

factory 必須回傳 BaseTool，而且回傳的 tool name 必須與 registry name 相同。

## 保留原有 Tools

原有 tools 不需要改寫成 skills：

    import akasha

    agent = akasha.agents(
        tools=[database_tool, calculator_tool],
        skills=["skills/research"],
    )

在這個例子中：

- database_tool 與 calculator_tool 從第一輪開始可用。
- research skill 的 instructions 需要等模型呼叫 load_skill 後才可用；database_tool 與 calculator_tool 從第一輪即可使用。

filesystem skill 不會從文件宣告 skill tool，因此不會產生文件層級的 tool name collision。若使用 Python Skill 綁定工具，仍會檢查 tool name collision。

## 多個 Skills

單一 agent 可以設定多個 skills：

    import akasha

    agent = akasha.agents(
        skills=[
            "skills/research",
            "skills/reviewer",
        ],
    )

模型可以依需求分別呼叫：

    load_skill(reference="skills/research")
    load_skill(reference="skills/reviewer")

每個 skill 載入後，該 skill 的 instructions 才會加入後續 model turn；工具仍由 agent 設定提供。

## Logs

如果啟用 logs，agent 會記錄設定的 skills；只有使用 Python Skill 綁定工具時，才會有 skill_tools，例如：

    {
        "skills": ["research"],
        "skill_versions": {
            "research": "1.0.0"
        },
        "skill_tools": {
            "research": ["web_search_tool"]
        }
    }

## 設計限制

目前 Phase 2b 聚焦於單一 agent 的動態 skill loading：

- 不包含 supervisor agent。
- 不包含 agent-to-agent handoff。
- 不包含 multi-agent routing。
- 不取代原有的 tools 參數。
- 不要求使用者自行管理 skill registry。
## Standard Metadata

Agent Skills 的 frontmatter 可以包含標準 optional metadata：

    ---
    name: research
    description: Research workflow
    license: MIT
    compatibility: Requires Python 3.11
    metadata:
      author: example-org
      version: "1.0"
    allowed-tools: Read
    ---

Akasha 會保存這些欄位到 Skill metadata，但目前不會根據 allowed-tools 自動注入或執行工具。

## Skill Resources

skill 可以包含額外的 resource directories：

    research/
    ├── SKILL.md
    ├── references/
    │   └── REFERENCE.md
    ├── assets/
    │   └── config.json
    └── scripts/
        └── inspect.py

SKILL.md 載入後，模型可以按需呼叫：

    read_skill_resource(
        skill="research",
        path="references/REFERENCE.md",
    )

resource path 必須是已載入 skill root 下的相對路徑。Akasha 會拒絕絕對路徑、.. traversal、skill root 外的檔案與 directory path。

第一版支援 UTF-8 文字檔，包含 Markdown、JSON、YAML、CSV 與 Python。binary assets 目前不會自動轉換或附加給模型。scripts 只可以被讀取，不會由 Akasha 自動執行；若應用程式需要執行腳本，必須透過既有 tools 明確提供執行能力。

預設單檔 resource 上限為 128 KiB，可以透過 agent 參數調整：

    agent = akasha.agents(
        skills=["skills/research"],
        max_resource_bytes=131072,
    )

read_skill_resource 只有在對應 skill 載入後才會出現在模型可用的 tools 中。

## Skill Script 執行

本機 filesystem skill 載入後，Akasha 會在內部提供 run_skill_script capability。使用者不需要另外設定 skill_executor 或註冊這個 tool：

    import akasha

    agent = akasha.agents(
        skills=["skills/research"],
    )

run_skill_script 會使用呼叫端主程式目前正在執行的環境。Python script 使用呼叫端的 Python interpreter，因此可以使用呼叫端已安裝、但 Akasha 本身沒有依賴的套件。

SKILL.md 應清楚說明 script 的相對路徑、使用時機、參數格式、輸出格式，以及需要的套件或外部程式：

    When the task requires normalization, run:
    scripts/read_data.py <value>
    Pass one argument and treat stdout as the normalized result.

Akasha 目前會：

- 只執行已載入 skill root 下的 script。
- 拒絕絕對路徑、.. traversal、directory path 與 symlink escape。
- 使用結構化 args，不接受任意 shell command 字串。
- 預設限制 script 執行時間與 stdout/stderr 輸出大小。
- 將 exit code、stdout、stderr 回傳給 agent。
- 不自動安裝套件、不修改呼叫端環境。

ls、mv、grep、write_file 與 edit_file 等通用 filesystem tools 不在本次範圍內；skill 只會按 SKILL.md 指定的相對路徑讀取或執行資源。
