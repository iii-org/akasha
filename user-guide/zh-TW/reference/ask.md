# `ask`

`ask` 是向模型提問最簡單的公開 API。

## 建立問答物件

```python
asker = akasha.ask(
    model="gemini:gemini-2.5-flash",
    temperature=0.0,
    system_prompt="請清楚且簡短地回答。",
)
```

常用建立參數：

| 參數 | 意義 |
| --- | --- |
| `model` | Provider 與模型別名，例如 `gemini:gemini-2.5-flash`。 |
| `temperature` | 取樣溫度；`0.0` 是偏向穩定的起點。 |
| `system_prompt` | 給模型的全域指示。 |
| `max_input_tokens` | 本次請求允許的最大輸入量。 |
| `max_output_tokens` | 允許產生的最大輸出量。 |
| `stream` | 是否使用串流輸出。 |
| `thinking` | 是否啟用 Provider 支援的推理內容。 |

## 提出問題

```python
import akasha

asker = akasha.ask(model="gemini:gemini-2.5-flash")
answer = asker("什麼是向量資料庫？")
print(answer)
```

呼叫方式：

```python
answer = asker(
    prompt,
    info="./docs",              # 可選：檔案、目錄、URL 或文件資料
    history_messages=[],         # 可選：之前的訊息
)
```

一般回傳值是完整的 `str`。串流行為請參考 [串流事件](../tutorials/streaming.md)。

!!! tip
    如果問題需要支援檔案或 URL，可以使用 `info`。如果需要可重複的文件檢索流程，請使用 `RAG`。
