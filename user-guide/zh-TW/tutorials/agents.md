# 建立 Agent 與使用工具

當模型需要在任務過程中決定是否呼叫工具時，可以使用 `akasha.agents()`。

## 最小 Agent 範例

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=False,
)

answer = agent("請解釋什麼時候適合使用工具呼叫 Agent。")
print(answer)
```

請將 `tools` 替換成應用程式允許的工具。工具應該有清楚的用途與安全的輸入邊界。

## Skills 與 MCP

Skills 可以從 Skill 目錄提供指示與允許使用的工具。MCP 工具可以先正規化，再傳給 `akasha.agents(tools=...)`。

!!! warning
    請把工具視為應用程式能力：驗證輸入、限制檔案與網路存取，也不要透過工具參數暴露秘密資訊。

想要逐步顯示輸出時，請搭配閱讀 [Streaming 事件](streaming.md)。
