# `RAG`

`RAG` combines document loading, embedding, retrieval, and answer generation.

## Create a RAG instance

```python
rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    chunk_size=1000,
    search_type="auto",
)
```

Common constructor options:

| Option | Meaning |
| --- | --- |
| `model` | Chat model used to generate the answer. |
| `embeddings` | Embedding model used for documents and queries. |
| `chunk_size` | Approximate size of document chunks. |
| `search_type` | Retrieval strategy, commonly `auto`. |
| `max_input_tokens` | Maximum input size for the final answer. |
| `use_chroma` | Use an existing Chroma-backed data source when applicable. |
| `stream` | Whether answer generation is streamed. |

## Ask about documents

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag(
    data_source="./docs",
    prompt="What are the main ideas in these documents?",
)
print(answer)
```

Call shape:

```python
answer = rag(
    data_source=["notes.md", "report.pdf"],
    prompt="Summarize the important findings.",
)
```

The normal return value is a final `str`. The first run may take longer because documents and embeddings need to be prepared.
