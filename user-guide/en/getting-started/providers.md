# Configure a model provider

Akasha selects providers through model aliases. Put credentials in your shell environment or a local `.env` file. Never commit API keys.

```env
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Examples of model aliases:

```text
gemini:gemini-2.5-flash
openai:gpt-4o
anthropic:claude-3-5-sonnet-latest
ollama:qwen3:8b
```

The provider name before `:` selects the integration. The model name after `:` selects the model.

## A safe configuration check

Do not print the key. Check only that the variable exists:

```python
import os

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Set GEMINI_API_KEY before running this example")

print("Provider configuration is present")
```

!!! warning
    A configured key may still incur provider charges. Check the provider's model access and billing settings before running live examples.

Next: [Run your first chat](first-chat.md).
