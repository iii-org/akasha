# 建立 Agent、Tool 與 Skill

這篇會從只有模型的 Agent 開始，接著加入一個應用程式 Tool，再加入一個 Skill。建議依照順序執行，每一步都先確認成功再繼續。

## 開始前準備

建立並啟用虛擬環境，再安裝 akasha：

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

設定聊天模型需要的 key。本篇使用 Gemini：

```powershell
$env:GEMINI_API_KEY = "your_key"
```

不要將真正的 key 寫入並提交到 Python 檔案。

## 第一步：建立只有模型的 Agent

建立 `agent_step1.py`：

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=False,
)

answer = agent("請用兩句話解釋什麼是工具呼叫 Agent。")
print(answer)
```

執行：

```bash
python agent_step1.py
```

此時 Agent 可以回答問題，但還不能執行應用程式動作。空的 `tools` 清單是刻意的設定。

## 第二步：加入 Tool

Tool 是模型可以選擇呼叫的操作。請給它清楚的名稱、準確的說明、具型別的參數，以及安全的執行邊界。

建立 `agent_step2.py`：

```python
import akasha


def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


add_tool = akasha.create_tool(
    "Add two integers. Use this when the user asks for an addition.",
    add_numbers,
    tool_name="add_numbers",
)

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[add_tool],
    stream=False,
    max_round=4,
)

answer = agent("請使用 add_numbers 工具計算 20 + 22。")
print(answer)
```

執行：

```bash
python agent_step2.py
```

模型會決定是否呼叫 `add_numbers`，實際計算則由 Python 函式完成。Tool 說明會成為模型指示的一部分，因此內容要具體且符合實際行為。

!!! warning
    不要把無限制的 Shell、檔案系統、資料庫或網路函式直接暴露成 Tool。請驗證參數，只允許應用程式真正需要的操作。

## 第三步：加入 Skill

Skill 是一個包含 `SKILL.md` 指示檔與可選資源或腳本的目錄，用來描述 Agent 何時以及如何使用某項能力。

Repository 已經有一個可執行的 Skill：

```text
examples/examples_skills/hello-skill/
├─ SKILL.md
└─ scripts/greet.py
```

在 repository 根目錄建立 `agent_step3.py`：

```python
from pathlib import Path

import akasha


skill_path = Path("examples/examples_skills/hello-skill").resolve()

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=[str(skill_path)],
    stream=False,
)

answer = agent(
    "請使用 hello-skill 向 Alice 打招呼，遵循 Skill 指示並回傳腳本輸出。"
)
print(answer)
```

從 repository 根目錄執行：

```bash
python agent_step3.py
```

Skill 不是 Tool 的替代品。Skill 提供指示、資源與受控制的工具能力；Agent 仍然需要遵循 Skill 宣告的工作流程。

## 第四步：觀察執行過程

如果應用程式需要在執行期間看到 thinking、tool 與 answer 事件，可以使用 `stream=True`：

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=[str(skill_path)],
    stream=True,
    thinking=True,
    verbose=True,
)

for event in agent("請使用 hello-skill 向 Alice 打招呼。"):
    if event["type"] == "tool":
        print("[tool]", event["data"])
    elif event["type"] == "thinking":
        print("[thinking]", event["data"])
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

事件意義請參考[串流事件](streaming.md)。

## 常見問題

- 找不到 `GEMINI_API_KEY`：在執行 Python 的同一個終端機設定環境變數。
- 找不到 Skill：從 repository 根目錄執行，或改用絕對 Skill 路徑。
- Tool 沒有被使用：確認 `create_tool()` 有回傳 Tool，並且傳入 `tools` 清單。
- Tool 行為不符合預期：改善函式的型別註記與說明，再把 Prompt 寫得更清楚。

Repository 的 `examples/examples_skills/` 中也有可執行的相關範例。
