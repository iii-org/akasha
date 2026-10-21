# `myTokenizer`

`myTokenizer` is the helper API for counting tokens with a model-specific tokenizer.

## `compute_tokens()`

```python
import akasha.helper as ah

count = ah.myTokenizer.compute_tokens(
    "Count this text.",
    "openai:gpt-4o",
)
print(count)
```

Signature:

```python
myTokenizer.compute_tokens(
    text,
    model_id,
    model_path="./tokenizers",
    save_tokenizer=True,
    *,
    exact=False,
)
```

| Argument | Meaning |
| --- | --- |
| `text` | Text to count. |
| `model_id` | Model alias, such as `openai:gpt-4o` or a Hugging Face model alias. |
| `model_path` | Local directory used for Hugging Face tokenizer files. |
| `save_tokenizer` | Whether to save a loaded Hugging Face tokenizer locally. |
| `exact` | Gemini only: explicitly call the provider's remote token counter. Default: `False`. |

## Supported model families

- OpenAI models use `tiktoken`.
- Gemini aliases (`gemini:`, `google:`, `gemi:`) use a local `cl100k_base` estimate by default. They do not send a separate Gemini token-count request.
- Hugging Face models load their tokenizer and may download it the first time.
- Other model aliases use a fallback tokenizer calculation.

Use the model alias that matches the actual request. The count is useful for input budgeting, but it is not a guarantee of the Provider's final usage or billing count.

To explicitly request Gemini's remote text count:

```python
count = ah.myTokenizer.compute_tokens(
    "Count this text.", "gemini:gemini-2.5-flash", exact=True,
)
```

This requires `GEMINI_API_KEY` and respects `GOOGLE_GENAI_USE_VERTEXAI`.
The counter uses a 10-second request timeout, one SDK attempt, and closes its
client. Errors propagate instead of silently returning a local estimate.
It counts the supplied text, not the complete formatted generation request.
`exact=True` is not supported for other model families.

The examples above are complete and can be copied into your own project directly.
