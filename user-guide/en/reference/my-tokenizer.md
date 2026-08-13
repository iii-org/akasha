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
)
```

| Argument | Meaning |
| --- | --- |
| `text` | Text to count. |
| `model_id` | Model alias, such as `openai:gpt-4o` or a Hugging Face model alias. |
| `model_path` | Local directory used for Hugging Face tokenizer files. |
| `save_tokenizer` | Whether to save a loaded Hugging Face tokenizer locally. |

## Supported model families

- OpenAI models use `tiktoken`.
- Gemini models use the Gemini token calculation when available, with a fallback estimate.
- Hugging Face models load their tokenizer and may download it the first time.
- Other model aliases use a fallback tokenizer calculation.

Use the model alias that matches the actual request. The count is useful for input budgeting, but it is not a guarantee of the Provider's final usage or billing count.

The longer project example is [`examples/helper/ex_token_count.py`](https://github.com/iii-org/akasha/blob/master/examples/helper/ex_token_count.py).
