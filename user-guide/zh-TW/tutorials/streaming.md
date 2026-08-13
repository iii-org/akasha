# 串流事件

Streaming 讓應用程式可以在 Agent 執行期間逐步顯示輸出。

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=True,
    thinking=True,
)

for event in agent("請解釋向量資料庫與 Embedding 模型的差異。"):
    if event["type"] == "thinking":
        print("[thinking]", event["data"])
    elif event["type"] == "tool":
        print("[tool]", event["data"])
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

| 事件類型 | 意義 |
| --- | --- |
| `answer` | 最終回答的一段文字。 |
| `thinking` | Provider 提供的推理內容，可能需要啟用才會出現。 |
| `tool` | Tool 或 Skill 的結果。 |

不同 Provider 不一定會產生所有事件類型。請依照事件類型分支處理，並安全地忽略未知事件。
