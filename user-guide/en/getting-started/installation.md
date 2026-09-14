# Installation

## Requirements

- Python 3.11 or 3.12
- A virtual environment
- A model provider account if you use a remote model

## Lightweight installation

Use the lightweight extra for service-backed chat models, remote embeddings,
Chroma-backed RAG, LLM reranking, and memory workflows:

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

With `light`, `reranker="llm"` asks a chat-model API to order the candidate
documents. It does not load Torch or reranker model weights in the Akasha
process. It keeps PDF, DOCX, CSV, Markdown, and plain-text loading, but does not
install Streamlit, MLflow, Unstructured, the PPTX parser, or FAISS.

Install only the additional feature you need:

```bash
uv pip install "akasha-terminal[light,ui]"         # Streamlit toy UI
uv pip install "akasha-terminal[light,tracking]"   # MLflow experiment tracking
uv pip install "akasha-terminal[light,documents]"  # PPTX and Unstructured parsing
uv pip install "akasha-terminal[light,faiss]"      # search_type="faiss"
```

Akasha checks an optional dependency only when its feature is selected, so a
light environment can still import the package. A missing extra raises
`OptionalDependencyError` before model inference or retrieval starts and shows
complete `uv add` and `pip install` commands. Directory loading is the only
exception: unsupported files are skipped with a warning if other files load,
while an all-skipped directory raises the same actionable error.

## Full installation

Use the full extra when you need local Hugging Face models, local embeddings,
BGE reranking, BERTScore, PEFT, GPTQ, or llama.cpp:

```bash
uv pip install "akasha-terminal[full]"
```

!!! note
    Start with `light` unless you specifically need local-model features. It is smaller and easier to set up.

| Capability | `light` | `full` |
| --- | --- | --- |
| Cloud, Ollama, vLLM, and OpenAI-compatible chat models | Yes | Yes |
| Remote embeddings and local Chroma | Yes | Yes |
| `reranker="llm"` | Yes | Yes |
| Local BGE reranker | No | Yes |
| Local Hugging Face, BERTScore, PEFT, GPTQ, and llama.cpp | No | Yes |
| Streamlit UI, MLflow, PPTX/Unstructured, and FAISS | Add the required extra | All included |

`full` remains the one-command compatibility profile. Along with the local
model backends, it includes the `ui`, `tracking`, `documents`, and `faiss`
feature stacks.

Every installation requires Python `>=3.11,<3.13` and NumPy `>=2,<3`.
Some packages in `full` contain native extensions. On Windows you may still
need Visual Studio C++ Build Tools or a prebuilt wheel index supplied by the
corresponding project.

## Verify the installation

```bash
python -c "import akasha; print('akasha imported successfully')"
```

Next: [Configure a model provider](providers.md).
