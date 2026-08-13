# `ask`

`ask` is the simplest public API for asking a model a question.

## Create an asker

```python
asker = akasha.ask(
    model="gemini:gemini-2.5-flash",
    temperature=0.0,
    system_prompt="Answer clearly and briefly.",
)
```

Common constructor options:

| Option | Meaning |
| --- | --- |
| `model` | Provider and model alias, such as `gemini:gemini-2.5-flash`. |
| `temperature` | Sampling temperature. `0.0` is a deterministic-oriented starting point. |
| `system_prompt` | Instructions applied to the model. |
| `max_input_tokens` | Maximum input size accepted by the request. |
| `max_output_tokens` | Maximum generated output size. |
| `stream` | Whether the call returns streamed output. |
| `thinking` | Whether supported thinking/reasoning output is enabled. |

## Ask a question

```python
import akasha

asker = akasha.ask(model="gemini:gemini-2.5-flash")
answer = asker("What is a vector store?")
print(answer)
```

Call shape:

```python
answer = asker(
    prompt,
    info="./docs",              # optional file, directory, URL, or document data
    history_messages=[],         # optional previous messages
)
```

The normal return value is a final `str`. For streaming behavior, see [Streaming events](../tutorials/streaming.md).

!!! tip
    Use `info` when the question needs supporting files or URLs. Use `RAG` when you need a repeatable document retrieval workflow.
