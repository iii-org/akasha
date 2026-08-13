# `summary`

`summary` 可以摘要文字、檔案、URL 或支援的文件物件。

## 建立摘要物件

```python
summarizer = akasha.summary(
    model="gemini:gemini-2.5-flash",
    sum_type="map_reduce",
    sum_len=500,
    chunk_size=500,
    chunk_overlap=50,
)
```

常用建立參數：

| 參數 | 意義 |
| --- | --- |
| `model` | 產生摘要的模型。 |
| `sum_type` | 摘要策略：`map_reduce` 或 `refine`。 |
| `sum_len` | 摘要的目標長度。 |
| `chunk_size` | 輸入分段大小。 |
| `chunk_overlap` | 相鄰分段的重疊大小。 |
| `temperature` | 取樣溫度。 |

## 摘要內容

```python
import akasha

summarizer = akasha.summary(model="gemini:gemini-2.5-flash")
summary = summarizer("Akasha 是一套文件感知 AI 應用程式的 Python 工具。")
print(summary)
```

`content` 也可以是檔案路徑、URL、來源清單或支援的文件物件。一般回傳值是完整的 `str`。
