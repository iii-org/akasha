# Configure a model provider

akasha selects providers through model aliases. Put credentials in your shell environment or a local `.env` file. Never commit API keys.

```env
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

## Gemini backend modes

For the Gemini Developer API, set only an API key:

```env
GEMINI_API_KEY=your_key
```

To use Vertex AI **Express Mode** with an API key instead of Application
Default Credentials (ADC), use the same key and explicitly opt in:

```env
GEMINI_API_KEY=your_key
GOOGLE_GENAI_USE_VERTEXAI=true
```

Do not set `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_LOCATION` for Express Mode.
They select the full Vertex AI project/location path, which requires ADC and
takes precedence over an API key. Akasha's Vertex API-key support is Express
Mode; it does not configure the full ADC-based Vertex AI path.

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
