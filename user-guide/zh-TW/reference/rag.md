# `RAG`

`RAG` 結合文件載入、Embedding、檢索與回答產生。

## 建立 RAG 物件

```python
rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    chunk_size=1000,
    search_type="auto",
)
```

常用建立參數：

| 參數 | 意義 |
| --- | --- |
| `model` | 產生回答的聊天模型。 |
| `embeddings` | 文件與問題使用的 Embedding 模型。 |
| `chunk_size` | 文件分段的大約大小。 |
| `search_type` | 檢索策略，常見值為 `auto`。 |
| `max_input_tokens` | 最終回答允許的最大輸入量。 |
| `use_chroma` | 在適用情況下使用既有的 Chroma 資料來源。 |
| `stream` | 是否使用串流產生回答。 |

## 詢問文件

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag(
    data_source="./docs",
    prompt="這些文件的主要內容是什麼？",
)
print(answer)
```

呼叫方式：

```python
answer = rag(
    data_source=["notes.md", "report.pdf"],
    prompt="請整理重要發現。",
)
```

一般回傳值是完整的 `str`。第一次執行可能較久，因為需要準備文件與 Embedding。
