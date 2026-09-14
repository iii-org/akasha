# `RAG`

`RAG` combines document loading, embedding, first-stage retrieval, optional second-stage reranking, and answer generation.

## Create a RAG instance

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

## Common constructor options

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | `openai:gpt-3.5-turbo` | Chat model that generates the final answer. |
| `embeddings` | `openai:text-embedding-ada-002` | Embedding model used for documents and queries. |
| `chunk_size` | `1000` | Approximate size of document chunks. |
| `search_type` | `auto` | First-stage retrieval strategy. |
| `reranker` | `None` | Optional second-stage reranker. |
| `reranker_model` | `gemini:gemini-2.5-flash` | Chat model used by `reranker="llm"`. |
| `rerank_top_k` | `5` | Number of documents retained after reranking. |
| `max_input_tokens` | `3000` | Maximum input size for the final answer. |
| `use_chroma` | `False` | Use an existing Chroma-backed data source when applicable. |
| `stream` | `False` | Whether final answer generation is streamed. |

`reranker_model` and `model` are independent settings. The default reranker model is loaded lazily only when `reranker="llm"` actually runs. Akasha does not create a Gemini client when reranking is disabled.

## Supported `reranker` values

| Value | Installation | Behavior |
| --- | --- | --- |
| `None` | base / `light` / `full` | Skips second-stage reranking; `rerank_top_k` has no effect. |
| `"llm"` | base / `light` / `full` | Sends the question and candidates to `reranker_model` and requests a complete ID ordering. |
| `"local"` or `"bge"` | `full` | Uses the default `BAAI/bge-reranker-base`. |
| `"local:<model>"` or `"bge:<model>"` | `full` | Uses the selected Hugging Face sequence-classification model. |
| Callable | base / `light` / `full` | Calls `(query, documents)` and verifies that it only reorders the original candidates. |

`rerank_top_k` must be a positive integer. Every reranker must first provide a complete ordering. Akasha validates that ordering before retaining the top N documents, so a callable cannot omit candidates itself.

## LLM reranker call flow

```text
search_type retrieves candidates
        ↓
reranker_model orders every candidate ID
        ↓
retain rerank_top_k documents
        ↓
model generates an answer from those documents
```

The LLM reranker is instructed to return:

```json
{"order": [2, 0, 1]}
```

Every candidate ID must appear exactly once. Invalid JSON, omitted IDs, or duplicate IDs raise `ValueError`. A cloud reranker also adds one model request and sends the candidate text to that provider.

## `light` example: default Gemini reranker

This example requires `GEMINI_API_KEY`:

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="llm",
)

answer = rag("./docs", "Summarize the most important maintenance risks.")
print(answer)
```

The two omitted defaults are:

```python
reranker_model="gemini:gemini-2.5-flash"
rerank_top_k=5
```

## `full` example: local BGE reranker

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="local:BAAI/bge-reranker-base",
    rerank_top_k=5,
)

answer = rag("./docs", "Which passages best support the document's conclusion?")
print(answer)
```

This reranker requires `akasha-terminal[full]`, Torch, and Transformers. The first use downloads the model.

## Custom callable example

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

## Ask about documents

```python
answer = rag(
    data_source=["notes.md", "report.pdf"],
    prompt="Summarize the important findings.",
)
```

The normal return value is a final `str`. Inspect `rag.docs` to see the documents and ordering that reached the answer model after reranking.

## Override one call

Both `__call__()` and `selfask_RAG()` accept keyword-only `reranker`, `reranker_model`, and `rerank_top_k` overrides:

```python
answer = rag(
    "./docs",
    "Answer using only the three most relevant documents.",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

For a complete workflow and operational tradeoffs, see [Build a RAG workflow](../tutorials/rag.md).
