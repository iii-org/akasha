# Build a RAG workflow

RAG retrieves relevant document content before asking the model to answer. akasha uses a chat model and an embedding model for different parts of this workflow.

## Complete example

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

Here:

- `model` generates the answer.
- `embeddings` converts document text and the question into vectors.
- `./docs` is the document source.

## Common problems

- The document path must exist and contain supported files.
- The embedding provider must be configured separately from the chat provider.
- The first run may take longer while documents and embeddings are prepared.

Next: use [Agents](agents.md) when the model needs to call tools instead of only retrieving documents.
