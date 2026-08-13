# Akasha 可以做什麼

Akasha 提供一組公開入口，對應常見的模型應用情境：

| 功能 | 入口 | 適合情境 |
| --- | --- | --- |
| 聊天與問答 | `akasha.ask()` | 想直接向模型提問，可選擇加入背景資料。 |
| 檢索增強生成 | `akasha.RAG()` | 需要先載入並搜尋文件，再產生回答。 |
| Agent | `akasha.agents()` | 需要工具呼叫或多步驟任務。 |
| 摘要 | `akasha.summary()` | 需要摘要文字、檔案或 URL。 |
| 長期記憶 | `MemoryManager` | 需要儲存與取回語意記憶。 |

公開 API 透過 `gemini:`、`openai:`、`anthropic:` 與 `ollama:` 等別名隱藏 Provider 建立細節。

## 一個實用的理解方式

```text
問題 → 可選的背景資料 → 模型 → 回答
                    ↘ 工具／檢索／記憶
```

不確定該用哪個入口時，請閱讀[選擇 API](choosing-an-api.md)。
