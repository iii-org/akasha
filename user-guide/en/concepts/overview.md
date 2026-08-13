# What Akasha provides

Akasha exposes a small set of public entry points for common model workflows:

| Capability | Entry point | Use it when... |
| --- | --- | --- |
| Chat and question answering | `akasha.ask()` | You need an answer from a model, optionally with context. |
| Retrieval-augmented generation | `akasha.RAG()` | You need to load and search documents before answering. |
| Agents | `akasha.agents()` | You need tool calling or a multi-step task. |
| Summaries | `akasha.summary()` | You need to summarize text, files, or URLs. |
| Long-term memory | `MemoryManager` | You need to store and retrieve semantic memories. |

The public API hides provider-specific construction behind aliases such as `gemini:`, `openai:`, `anthropic:`, and `ollama:`.

## A useful mental model

```text
Question → optional context → model → answer
                         ↘ tools / retrieval / memory
```

Read [Choosing an API](choosing-an-api.md) when you are unsure which entry point to use.
