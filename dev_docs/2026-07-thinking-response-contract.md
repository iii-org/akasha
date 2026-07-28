# Thinking / Tool / Skill / Answer 回傳格式

Date: 2026-07-28

## 目的

本文件定義前台使用 `akasha.ask()` 與 `akasha.agents()` 時，如何分辨：

- `thinking`：模型產生的思考摘要（不是最終答案）
- `tool`：工具執行結果；skill 也是透過這一類事件回傳
- `answer`：模型提供給使用者的最終回答內容

前台應依事件的 `type` 判斷用途，不要依文字內容猜測。

## 快速判斷

| API / 設定 | 回傳形式 | 前台處理方式 |
|---|---|---|
| `ask(stream=False)` | `str` | 直接顯示，這就是答案 |
| `ask(stream=True, thinking=False)` | `str` chunks | 逐段串接後顯示為答案 |
| `ask(stream=True, thinking=True)` | event dict | 處理 `thinking` 與 `answer` |
| `agents(stream=False)` | `str` | 直接顯示，這就是答案 |
| `agents(stream=True)` | event dict | 處理 `thinking`、`tool`、`answer` |

## Streaming event 共通格式

### Thinking

```json
{
  "type": "thinking",
  "data": "我先整理問題中的條件。"
}
```

`data` 是 provider 提供的 thinking/reasoning 摘要。它可能會分成多個 chunk，前台若要顯示完整內容，應依順序串接同一個 stream 中所有 `thinking.data`。

這不是使用者最終答案；通常應放在可展開的「思考過程」區域。若產品不希望顯示思考摘要，可以直接忽略此事件。

### Answer

```json
{
  "type": "answer",
  "data": "根據條件，答案是 42。"
}
```

`answer.data` 可能是部分文字。前台應依順序串接所有 `answer.data`，串接完成後才是完整答案：

```python
answer = "".join(
    event["data"]
    for event in events
    if event.get("type") == "answer"
)
```

### Tool / Skill result

```json
{
  "type": "tool",
  "data": {
    "type": "tool",
    "name": "python_execute",
    "tool_call_id": "call-python-1",
    "content": "execution: repl\nstdout:\n5"
  }
}
```

`tool` event 代表工具已執行並回傳結果。前台可使用 `data.name` 顯示工具名稱，使用 `data.content` 顯示執行結果或放入工具細節區域。

目前 `agents(stream=True)` 回傳的是工具結果 `ToolMessage`，不是模型發出的完整 tool-call 參數。因此前台不應假設 `data` 一定包含 `args`；若需要完整呼叫參數，應讀取 `keep_logs` 紀錄或使用 verbose/log trace。

## Skill 如何辨識

Skill 沒有獨立的 event type。Skill 內部工具仍然會以：

```json
{
  "type": "tool",
  "data": {
    "name": "load_skill",
    "content": "Skill 'python-repl-skill' loaded."
  }
}
```

或：

```json
{
  "type": "tool",
  "data": {
    "name": "python_execute",
    "content": "execution: repl\nstdout:\n5"
  }
}
```

前台可依 `data.name` 判斷：

| `data.name` | 意義 |
|---|---|
| `load_skill` | 載入 skill；可顯示「正在準備技能」 |
| `read_skill_resource` | 讀取 skill 文件或資源 |
| `python_execute` | 執行 skill 指定或模型產生的 Python |
| 其他名稱 | 一般工具；依工具名稱或產品設定顯示 |

不要把所有 `tool` 都顯示成「skill」；只有工具名稱或產品 metadata 明確表示時才標記為 skill。

## `agents()` 完整範例

### 建立 agent

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=["examples/examples_skills/python-repl-skill"],
    stream=True,
    thinking=True,
)
```

### 可能的事件順序

```python
[
    {"type": "thinking", "data": "我需要先建立可重複使用的計算環境。"},
    {
        "type": "tool",
        "data": {
            "type": "tool",
            "name": "load_skill",
            "content": "Skill 'python-repl-skill' loaded.",
        },
    },
    {
        "type": "tool",
        "data": {
            "type": "tool",
            "name": "python_execute",
            "content": "execution: repl\nstdout:\n5.0",
        },
    },
    {"type": "answer", "data": "平均值是 5.0。"},
]
```

實際 stream 可能包含多個 thinking/answer chunk，也可能沒有 thinking 事件；前台必須接受事件數量為零個或多個。

### 前台事件分流範例

```python
thinking_parts = []
answer_parts = []

for event in agent("計算平均值"):
    event_type = event.get("type")

    if event_type == "thinking":
        thinking_parts.append(event.get("data", ""))
        render_thinking(event.get("data", ""))
    elif event_type == "tool":
        tool = event.get("data", {})
        render_tool(
            name=tool.get("name", "unknown"),
            result=tool.get("content", ""),
        )
    elif event_type == "answer":
        answer_parts.append(event.get("data", ""))
        render_answer(event.get("data", ""))

final_answer = "".join(answer_parts)
```

## `ask()` 範例

`ask(stream=True, thinking=True)` 使用相同的 `thinking` / `answer` event 格式，但不會產生 agent tool/skill event：

```python
qa = akasha.ask(
    model="gemini:gemini-2.5-flash",
    stream=True,
    thinking=True,
)

for event in qa("請解釋這個結果"):
    if event["type"] == "thinking":
        render_thinking(event["data"])
    elif event["type"] == "answer":
        render_answer(event["data"])
```

如果 `ask(stream=True, thinking=False)`，目前回傳的是純文字 chunks：

```python
for text_chunk in qa("請簡短回答"):
    render_answer(text_chunk)
```

這個模式沒有 `type` 欄位，不可套用 event dict 的分流程式。

