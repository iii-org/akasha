# Installation

## Requirements

- Python 3.11 or 3.12
- A virtual environment
- A model provider account if you use a remote model

## Lightweight installation

Use the lightweight extra for remote chat models, remote embeddings, Chroma-backed RAG, and memory workflows:

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

## Full installation

Use the full extra when you need local Hugging Face models, local embeddings, reranking, or other local ML features:

```bash
uv pip install "akasha-terminal[full]"
```

!!! note
    Start with `light` unless you specifically need local-model features. It is smaller and easier to set up.

## Verify the installation

```bash
python -c "import akasha; print('Akasha imported successfully')"
```

Next: [Configure a model provider](providers.md).
