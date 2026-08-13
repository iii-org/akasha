# `MemoryManager`

`MemoryManager` stores useful information from conversations and searches it later using semantic retrieval.

## Create a memory manager

```python
import akasha

memory = akasha.MemoryManager(
    memory_name="assistant",
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    memory_dirname="docs",
)
```

The memory files are stored below `memory_dirname / memory_name`.

## Add and search memory

```python
memory.add_memory(
    user_prompt="I prefer short technical explanations.",
    ai_response="I will keep future explanations concise.",
    language="en",
)

matches = memory.search_memory("What is my explanation preference?", top_k=3)
for item in matches:
    print(item)
```

Useful methods:

| Method | Purpose |
| --- | --- |
| `add_memory(user_prompt, ai_response, language="ch")` | Extract and store salient information from a conversation turn. |
| `search_memory(query, top_k=3)` | Return memories relevant to a query. |
| `show_memory(num=100)` | Return stored memory entries for inspection. |

Memory storage creates or updates local files and a vector store. Choose the memory directory deliberately and exclude private memory data from public repositories.
