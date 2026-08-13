# `handle_embeddings`

`akasha.helper.handle_embeddings()` 可以根據 Provider 別名、既有 Embedding 物件或自訂 Embedding function 建立 Embedding 物件。

## 建立 Embedding 物件

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings(
    "openai:text-embedding-3-small"
)
```

回傳的物件遵循 LangChain 的 `Embeddings` 介面。

## 可用方法

```python
query_vector = embedding_model.embed_query("什麼是檢索？")
document_vectors = embedding_model.embed_documents(
    ["第一份文件", "第二份文件"]
)
```

| 方法 | 回傳值 |
| --- | --- |
| `embed_query(text)` | 一個由浮點數組成的向量清單。 |
| `embed_documents(texts)` | 向量清單，每段輸入文字對應一個向量。 |

## Provider 別名

常見別名包括：

```text
openai:text-embedding-3-small
gemini:gemini-embedding-001
hf:BAAI/bge-base-en-v1.5
```

本機 Hugging Face Embedding 可能需要完整安裝；遠端 Provider 則需要設定對應的環境變數。

!!! note
    向量維度由選用的 Embedding 模型決定。請儲存並比較相容模型產生的向量，不要直接混用不同向量空間的結果。
