# `agents`

`agents` 建立一個可以呼叫工具的 Agent。Agent 只能使用你明確提供的工具。

## 建立 Agent

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    skills=None,
    stream=False,
)
```

常用建立參數：

| 參數 | 意義 |
| --- | --- |
| `tools` | Agent 可以呼叫的工具或工具清單。 |
| `skills` | 擴充 Agent 的 Skill 路徑或 Skill 物件。 |
| `model` | Provider 與模型別名。 |
| `max_round` | Agent 與工具互動的最大輪數。 |
| `max_past_observation` | Agent 保留的先前觀察數量。 |
| `stream` | 是否回傳串流事件而非單一結果。 |
| `thinking` | 是否啟用 Provider 支援的推理內容。 |
| `max_resource_bytes` | Skill 工具可以讀取的資源大小限制。 |

## 呼叫 Agent

```python
answer = agent("請解釋什麼時候適合使用工具呼叫 Agent。")
print(answer)
```

非串流呼叫會回傳完整回答；非同步版本是：

```python
answer = await agent.acall("請非同步執行這個任務。")
```

使用 `stream=True` 時，呼叫會產生事件字典。請參考 [串流事件](../tutorials/streaming.md)。

!!! warning
    Tool 是應用程式能力。請驗證輸入，並限制檔案、網路與憑證的存取範圍。
