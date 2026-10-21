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

## Empty single-pass answers

Non-streaming single-pass text generation raises `EmptyResponseError` when the
provider returns no visible text. Akasha does not automatically regenerate the
answer. This also applies to other APIs using the shared `call_model` helper;
provider SDK retries remain separate. Batch and streaming helpers retain their
existing behavior.

```python
from akasha.helper import EmptyResponseError

try:
    answer = asker("What is a vector store?")
except EmptyResponseError as error:
    print(error.diagnostics)
```

Diagnostics include recognized finish/block reasons, available token counts,
content types, tool-call count, and attempt count. They exclude answer text,
reasoning, tool arguments, and arbitrary provider metadata. A thinking-only or
tool-only response is not a text answer. Visible partial text is still returned
when the provider reports truncation.

!!! tip
    Use `info` when the question needs supporting files or URLs. Use `RAG` when you need a repeatable document retrieval workflow.
