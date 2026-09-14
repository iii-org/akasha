# `RAG`

`RAG` 結合文件載入、Embedding、第一階段檢索、可選的第二階段重排，以及回答產生。

## 建立 RAG 物件

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    chunk_size=1000,
    search_type="auto",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=5,
)
```

## 常用建立參數

| 參數 | 預設值 | 意義 |
| --- | --- | --- |
| `model` | `openai:gpt-3.5-turbo` | 產生最終回答的聊天模型。 |
| `embeddings` | `openai:text-embedding-ada-002` | 文件與問題使用的 Embedding 模型。 |
| `chunk_size` | `1000` | 文件分段的大約大小。 |
| `search_type` | `auto` | 第一階段檢索策略。 |
| `reranker` | `None` | 可選的第二階段 reranker。 |
| `reranker_model` | `gemini:gemini-2.5-flash` | `reranker="llm"` 使用的聊天模型。 |
| `rerank_top_k` | `5` | 重排後保留的文件數量。 |
| `max_input_tokens` | `3000` | 最終回答允許的最大輸入量。 |
| `use_chroma` | `False` | 在適用情況下使用既有的 Chroma 資料來源。 |
| `stream` | `False` | 是否串流產生最終回答。 |

`reranker_model` 與 `model` 是兩個獨立設定。預設 reranker model 只會在 `reranker="llm"` 實際執行時 lazy loading；沒有啟用 reranker 時，不會建立 Gemini client。

## `reranker` 可用值

| 值 | 安裝需求 | 行為 |
| --- | --- | --- |
| `None` | base／`light`／`full` | 不執行第二階段重排，`rerank_top_k` 不生效。 |
| `"llm"` | base／`light`／`full` | 將問題與候選文件送給 `reranker_model`，要求回傳完整 ID 排序。 |
| `"local"` 或 `"bge"` | `full` | 使用預設 `BAAI/bge-reranker-base`。 |
| `"local:<model>"` 或 `"bge:<model>"` | `full` | 使用指定的 Hugging Face sequence-classification model。 |
| Callable | base／`light`／`full` | 呼叫 `(query, documents)` 並驗證它只重排原候選文件。 |

`rerank_top_k` 必須是正整數。所有 reranker 都必須先提供完整排序；Akasha 驗證後才截取前 N 筆，因此 callable 不可以自行漏掉候選文件。

## LLM reranker 的呼叫流程

```text
search_type 找出候選文件
        ↓
reranker_model 排列所有候選 ID
        ↓
只保留 rerank_top_k 筆
        ↓
model 根據保留的文件產生回答
```

LLM reranker 會要求模型回傳：

```json
{"order": [2, 0, 1]}
```

每個候選 ID 必須恰好出現一次。格式錯誤或 ID 遺漏／重複會產生 `ValueError`。Cloud reranker 也會增加一次模型請求，並將候選文字傳給該 Provider。

## `light` 範例：預設 Gemini reranker

需要 `GEMINI_API_KEY`：

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="llm",
)

answer = rag("./docs", "請整理最重要的維護風險。")
print(answer)
```

省略的兩個預設值是：

```python
reranker_model="gemini:gemini-2.5-flash"
rerank_top_k=5
```

## `full` 範例：本機 BGE reranker

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="local:BAAI/bge-reranker-base",
    rerank_top_k=5,
)

answer = rag("./docs", "哪幾段最能支持文件的結論？")
print(answer)
```

這個 reranker 需要 `akasha-terminal[full]`、Torch 與 Transformers；第一次使用會下載模型。

## 自訂 callable 範例

```python
def reverse_reranker(query, documents):
    return list(reversed(documents))


rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    reranker=reverse_reranker,
    rerank_top_k=2,
)
```

## 詢問文件

```python
answer = rag(
    data_source=["notes.md", "report.pdf"],
    prompt="請整理重要發現。",
)
```

一般回傳值是完整的 `str`。重排後實際提供給回答模型的文件可從 `rag.docs` 查看。

## 單次覆寫

`__call__()` 與 `selfask_RAG()` 都接受 keyword-only 的 `reranker`、`reranker_model` 與 `rerank_top_k`：

```python
answer = rag(
    "./docs",
    "只使用最相關的三筆文件回答。",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

更多完整流程與限制請閱讀 [建立 RAG 流程](../tutorials/rag.md)。
