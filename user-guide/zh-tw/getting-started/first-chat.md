# 第一個 `ask` 對話

`ask` 是將問題送給模型最簡單的方式。

## 完整範例

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa("請用一段文字解釋什麼是檢索增強生成？")
print(answer)
```

未啟用 streaming 時，呼叫會回傳一個完整字串：

```text
檢索增強生成會先擷取相關文件，再將文件內容交給模型產生回答……
```

實際文字會依模型而不同，應使用回傳值，而不是比對整段固定文字。

## 下一步怎麼選

- 有文件要搜尋？請閱讀 [RAG](../tutorials/rag.md)。
- 需要工具或多步驟行動？請閱讀 [Agent](../tutorials/agents.md)。
- 需要模型產生內容時就逐步顯示？請閱讀 [Streaming 事件](../tutorials/streaming.md)。

!!! tip
    第一個範例保持簡單，再一次加入一種 Prompt、文件或工具，這樣比較容易找出問題。
