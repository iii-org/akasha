# 建立 RAG 流程

RAG 會先找出與問題相關的文件內容，再把內容交給聊天模型產生回答。本篇會說明第一階段檢索、第二階段重排，以及 `light` 和 `full` 可使用的 reranker。

## 學習目標

完成本篇後，你會知道如何：

- 使用 `search_type` 找出候選文件。
- 使用 `reranker` 對候選文件重新排序。
- 用 `rerank_top_k` 控制最後送進回答 prompt 的文件數量。
- 分別設定回答模型與 `reranker_model`。
- 在 `light` 使用 LLM reranker，或在 `full` 使用本機 BGE reranker。

## 前置條件

LLM reranker 可使用 `light`：

```bash
uv pip install "akasha-terminal[light]"
```

以下 Gemini 範例是 live example，會呼叫外部服務且可能產生費用。請先在環境變數或 `.env` 設定：

```env
GEMINI_API_KEY=your_key
```

文件目錄必須存在，例如將一個或多個 `.txt`、`.md` 或 PDF 文件放在 `./docs`。

## 最小完整範例

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag(
    "./docs",
    "這些文件的主要內容是什麼？",
)
print(answer)
```

這個版本不設定 `reranker`，流程只有文件切分、Embedding、第一階段檢索與回答產生；`rerank_top_k` 不會截斷檢索結果。

## 檢索與重排的差異

| 設定 | 所在階段 | 用途 |
| --- | --- | --- |
| `search_type` | 第一階段檢索 | 從向量庫或其他 retriever 找出候選文件。 |
| `reranker` | 第二階段重排 | 根據原始問題重新排列候選文件。 |
| `reranker_model` | LLM 重排 | 指定負責排序的聊天模型；預設是 `gemini:gemini-2.5-flash`。 |
| `rerank_top_k` | 重排完成後 | 只保留最相關的前 N 筆；預設為 `5`。 |

推薦把兩個階段明確分開，例如 `search_type="auto"` 搭配 `reranker="llm"` 或本機 BGE，而不是把新 reranker 與歷史的 `search_type` 名稱混在一起。

## `light`：使用 LLM API 重排

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",  # 預設值
    rerank_top_k=5,                             # 預設值
)

answer = rag("./docs", "請找出與維護成本最相關的說明。")
print(answer)
```

執行時會發生兩次聊天模型呼叫：

1. `reranker_model` 收到問題與候選文件，回傳候選文件 ID 的排序。
2. `model` 收到排序後保留的前五筆文件，產生最終回答。

`reranker_model` 是 lazy loading；只有實際使用 `reranker="llm"` 時才會建立。因此只建立 `RAG` 物件，或完全不啟用 reranker，都不需要 Gemini key。

!!! warning
    使用 cloud LLM reranker 時，候選文件內容會額外傳送給該 Provider 一次，並增加延遲與 token 費用。即使最終回答使用 streaming，rerank 仍會先以非串流方式完成。

### 回答模型與 reranker 模型可以不同

例如由本機 Ollama 產生答案，但由 Gemini 排序：

```python
rag = akasha.RAG(
    model="ollama:qwen3:8b",
    embeddings="gemini:gemini-embedding-001",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

這個範例仍需要 `GEMINI_API_KEY`，也需要 Ollama service 已啟動並具有 `qwen3:8b` 模型。

## `full`：使用本機 BGE 重排

先安裝完整版本：

```bash
uv pip install "akasha-terminal[full]"
```

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="local:BAAI/bge-reranker-base",
    rerank_top_k=5,
)

answer = rag("./docs", "請找出最直接支持結論的段落。")
print(answer)
```

第一次執行會從 Hugging Face 下載 BGE 模型。排序在 Akasha process 中透過 Torch 執行，候選文件不必為了 rerank 傳給另一個 cloud LLM。也可以使用 `reranker="local"` 或 `reranker="bge"` 取得相同的預設 BGE 模型。

!!! note
    BERTScore 是回答／參考文字的語意評估指標，不是這條 RAG 流程中的第二階段 reranker；它同樣屬於 `full` 的本機模型功能。

## 自訂 reranker

Callable 必須接收 `(query, documents)`，並將每一筆候選文件恰好回傳一次。Akasha 驗證完整排序後，再套用 `rerank_top_k`。

```python
import akasha


def keyword_reranker(query, documents):
    query_words = set(query.lower().split())
    return sorted(
        documents,
        key=lambda document: sum(
            word in document.page_content.lower() for word in query_words
        ),
        reverse=True,
    )


rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    reranker=keyword_reranker,
    rerank_top_k=5,
)

print(rag("./docs", "維護成本如何估算？"))
```

## 單次呼叫覆寫

建構子提供預設設定，每次呼叫也可以覆寫，不必建立新的 RAG 物件：

```python
answer = rag(
    "./docs",
    "只保留最相關的三段內容。",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

## 預期結果

一般呼叫回傳完整的 `str`。答案文字會依文件與 Provider 而改變；可以查看 `rag.docs`，確認本次實際保留的文件順序與數量。

## 常見問題

- `can not find the GEMINI_API_KEY`：預設 LLM reranker 使用 Gemini；設定 `GEMINI_API_KEY`，或把 `reranker_model` 改成已設定的其他 Provider／模型物件。
- 本機 BGE 出現缺少 Torch 或 Transformers：安裝 `akasha-terminal[full]`，不要只安裝 `light`。
- LLM 沒有回傳合法排序：reranker 必須產生 `{"order": [id, ...]}`，且每個候選 ID 恰好出現一次；否則 Akasha 會丟出 `ValueError`。
- `rerank_top_k` 沒有效果：必須同時設定 `reranker`；沒有第二階段重排時，Akasha 保留第一階段選出的文件。
- 文件路徑錯誤：確認路徑存在，且包含支援的檔案格式。

如果模型需要呼叫工具，而不只是搜尋文件，請接著閱讀 [Agent](agents.md)。
