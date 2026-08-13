# First chat with `ask`

The `ask` API is the simplest way to send a question to a model.

## Complete example

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa("Explain retrieval-augmented generation in one paragraph.")
print(answer)
```

The call returns a final string when streaming is disabled:

```text
Retrieval-augmented generation combines document retrieval with generation...
```

The exact wording depends on the selected model. Your application should use the returned value rather than compare the complete text literally.

## Choosing the next feature

- Have documents to search? Start with [RAG](../tutorials/rag.md).
- Need tools or multi-step actions? Read [Agents](../tutorials/agents.md).
- Need output while the model is working? Read [Streaming events](../tutorials/streaming.md).

!!! tip
    Keep the first example small. Add custom prompts, documents, and tools one at a time so that failures are easier to diagnose.
