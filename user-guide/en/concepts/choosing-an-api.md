# Choosing an API

Start with the smallest API that matches your task.

## Decision table

| Your task | Start with |
| --- | --- |
| Ask a model a question | `akasha.ask()` |
| Ask questions about an image | `asker.vision()` |
| Generate a new image | `akasha.gen_image()` |
| Edit an existing image | `akasha.edit_image()` |
| Answer questions about local documents | `akasha.RAG()` |
| Let the model call tools | `akasha.agents()` |
| Summarize a file or URL | `akasha.summary()` |
| Remember facts across conversations | `MemoryManager` |

## Example: the same question with and without retrieval

```python
import akasha

# General model knowledge
qa = akasha.ask(model="gemini:gemini-2.5-flash")
print(qa("What is a vector store?"))

# Your documents as the source of context
rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)
print(rag("./docs", "What does our project say about vector stores?"))
```

If the answer depends on your files, use RAG instead of placing the entire file into a prompt yourself.
