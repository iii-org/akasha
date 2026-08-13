# Count tokens and control input length

Tokens are the units a model uses to process text. The number of tokens is not the same as the number of characters or words, and it depends on the selected model and tokenizer.

## Count tokens directly

```python
import akasha.helper as ah

text = "This is a short example for token counting."
tokens = ah.myTokenizer.compute_tokens(
    text,
    "openai:gpt-4o",
)

print(f"Token count: {tokens}")
```

Use the same model alias that you plan to use for the request. Different model families can produce different counts for the same text.

## Control an `ask` request

`ask`, `RAG`, `summary`, and related APIs count input tokens internally. Use `max_input_tokens` to limit the input accepted by a request:

```python
import akasha

qa = akasha.ask(
    model="gemini:gemini-2.5-flash",
    max_input_tokens=3000,
)

answer = qa(
    prompt="Summarize the important points.",
    info="./docs",
)
print(answer)
```

When the prompt or context is too large, Akasha may split or truncate document context, or raise an input-size error depending on the workflow.

## See token information in logs

Enable logs or verbose output when diagnosing input size:

```python
qa = akasha.ask(
    model="openai:gpt-4o",
    max_input_tokens=3000,
    keep_logs=True,
    verbose=True,
)
```

The logs can include prompt and document token information for the request.

!!! warning
    A local token count is an estimate of the tokenizer calculation. It may differ from the Provider's final usage or billing count, especially when request metadata, system messages, or provider-specific formatting are included.
