# Build a RAG workflow

RAG retrieves document content related to a question and then gives that content to a chat model. This tutorial explains first-stage retrieval, second-stage reranking, and the rerankers available in `light` and `full`.

## Learning goals

After completing this tutorial, you will know how to:

- Use `search_type` to retrieve candidate documents.
- Use `reranker` to reorder those candidates.
- Use `rerank_top_k` to control how many documents reach the answer prompt.
- Configure the answer model separately from `reranker_model`.
- Use an LLM reranker with `light` or a local BGE reranker with `full`.

## Prerequisites

An LLM reranker works with `light`:

```bash
uv pip install "akasha-terminal[light]"
```

The Gemini examples below are live examples. They call an external service and may incur charges. First configure the following environment variable or add it to a local `.env` file:

```env
GEMINI_API_KEY=your_key
```

The document directory must exist. For example, put one or more `.txt`, `.md`, or PDF files in `./docs`.

## Minimal complete example

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag(
    "./docs",
    "What are the main ideas in these documents?",
)
print(answer)
```

This version does not configure a `reranker`. Its stages are document splitting, embedding, first-stage retrieval, and answer generation. `rerank_top_k` does not truncate the retrieval results in this case.

## Retrieval versus reranking

| Setting | Stage | Purpose |
| --- | --- | --- |
| `search_type` | First-stage retrieval | Finds candidate documents through the vector store or another retriever. |
| `reranker` | Second-stage reranking | Reorders the candidates against the original question. |
| `reranker_model` | LLM reranking | Selects the chat model that performs ranking; the default is `gemini:gemini-2.5-flash`. |
| `rerank_top_k` | After reranking | Keeps only the top N documents; the default is `5`. |

Keep the two stages explicit. For example, use `search_type="auto"` with `reranker="llm"` or a local BGE reranker instead of mixing the new reranker with historical `search_type` names.

## `light`: rerank through an LLM API

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    search_type="auto",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",  # default
    rerank_top_k=5,                             # default
)

answer = rag("./docs", "Find the guidance most relevant to maintenance cost.")
print(answer)
```

This workflow makes two chat-model calls:

1. `reranker_model` receives the question and candidates, then returns an ordering of candidate IDs.
2. `model` receives the top five documents after reranking and generates the final answer.

`reranker_model` is loaded lazily. Akasha creates it only when `reranker="llm"` is actually used. Constructing a RAG instance, or not enabling a reranker, does not require Gemini credentials.

!!! warning
    A cloud LLM reranker sends the candidate document text to that provider one additional time. This adds latency and token cost. Even when the final answer is streamed, reranking completes as a non-streaming call first.

### The answer and reranker models can differ

For example, generate the answer through local Ollama while Gemini performs the ranking:

```python
rag = akasha.RAG(
    model="ollama:qwen3:8b",
    embeddings="gemini:gemini-embedding-001",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

This example still requires `GEMINI_API_KEY`, and the Ollama service must be running with the `qwen3:8b` model available.

## `full`: rerank with a local BGE model

Install the full profile first:

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

answer = rag("./docs", "Find the passages that most directly support the conclusion.")
print(answer)
```

The first run downloads the BGE model from Hugging Face. Ranking runs with Torch in the Akasha process, so the candidates do not have to be sent to a second cloud LLM for reranking. `reranker="local"` and `reranker="bge"` select the same default BGE model.

!!! note
    BERTScore is a semantic evaluation metric for answer/reference text. It is not the second-stage reranker in this RAG workflow, although it is also a local-model feature in `full`.

## Custom reranker

A callable must accept `(query, documents)` and return every candidate document exactly once. Akasha validates the complete ordering and then applies `rerank_top_k`.

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

print(rag("./docs", "How is maintenance cost estimated?"))
```

## Override settings for one call

The constructor provides reusable defaults. A call can override them without creating another RAG instance:

```python
answer = rag(
    "./docs",
    "Keep only the three most relevant passages.",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",
    rerank_top_k=3,
)
```

## Expected result

A normal call returns a final `str`. The wording depends on the documents and provider. Inspect `rag.docs` to verify the order and number of documents retained for the latest call.

## Common problems

- `can not find the GEMINI_API_KEY`: the default LLM reranker uses Gemini. Set `GEMINI_API_KEY`, or override `reranker_model` with another configured provider or model object.
- Missing Torch or Transformers for local BGE: install `akasha-terminal[full]`, not only `light`.
- Invalid LLM ordering: the reranker must produce `{"order": [id, ...]}` with every candidate ID exactly once. Akasha raises `ValueError` otherwise.
- `rerank_top_k` has no effect: configure `reranker` as well. Without second-stage reranking, Akasha preserves the documents selected by first-stage retrieval.
- Invalid document path: verify that the path exists and contains a supported file type.

Next: use [Agents](agents.md) when the model needs to call tools instead of only retrieving documents.
