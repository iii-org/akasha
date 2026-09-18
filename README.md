# akasha

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)

akasha is a Python toolkit for document question answering, image understanding, image generation and editing, retrieval-augmented generation (RAG), native tool-calling agents, summaries, and long-term semantic memory.

It provides one consistent interface for remote and local model workflows while keeping provider-specific integrations behind model aliases such as `openai:`, `gemini:`, `anthropic:`, and `ollama:`.

- manual: <https://iii-org.github.io/akasha/>
- Current package version: `1.8.0`

## What akasha provides

**Chat / QA** — Use `akasha.ask()` to ask a model a question, optionally with documents or web information.

**Vision** — Use `asker.vision()` to ask questions about one or more images and receive a text answer.

**Image generation** — Use `akasha.gen_image()` to generate a new image from a text prompt.

**Image editing** — Use `akasha.edit_image()` to remove, add, or change content in an existing image.

**Agents** — Use `akasha.agents()` for LangChain-native tool calling, streaming, thinking events, Skills, and MCP tools.

**RAG** — Use `akasha.RAG()` to load documents, create embeddings, search Chroma, and generate an answer.

**Summaries** — Use `akasha.summary()` to summarize text, files, or URLs with `map_reduce` or `refine`.

**Long-term memory** — Use `MemoryManager` to store and retrieve semantic memories with Chroma.

## Installation

Python 3.11 or 3.12 is recommended.

### Lightweight installation

Use `light` for service-backed chat models, remote embeddings, and LLM reranking:

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

`light` keeps Chroma-backed RAG and memory workflows. It can use cloud APIs,
Ollama, vLLM, or another OpenAI-compatible service, but it does not load model
weights in the Akasha process. It includes PDF, DOCX, CSV, Markdown, and text
loading, but leaves the UI, MLflow tracking, PPTX/Unstructured parsing, and
FAISS backend out of the default environment.

Add only the optional feature you need:

```bash
uv pip install "akasha-terminal[light,ui]"         # Streamlit toy UI
uv pip install "akasha-terminal[light,tracking]"   # MLflow experiment tracking
uv pip install "akasha-terminal[light,documents]"  # PPTX and Unstructured parsing
uv pip install "akasha-terminal[light,faiss]"      # search_type="faiss"
```

Optional features are checked only when selected, so a light installation can
always import Akasha. A missing extra raises `OptionalDependencyError` before
model inference or retrieval starts and shows both `uv add` and `pip install`
commands. Directory imports are the exception: unsupported files are skipped
with a warning when other files load successfully, but an all-skipped directory
raises the same actionable error.

### Full installation

Use `full` when you also need in-process Hugging Face embeddings/models, local
BGE reranking, BERTScore, PEFT, GPTQ, or llama.cpp:

```bash
uv pip install "akasha-terminal[full]"
```

The practical difference is:

| Installation | Chat models | Embeddings | Vector store | Local ML / rerank |
| --- | --- | --- | --- | --- |
| `light` | Cloud, Ollama, vLLM, OpenAI-compatible | Remote APIs | Local Chroma | LLM reranker |
| `full` | Everything in `light`, plus in-process HF | Remote and local | Local Chroma | BGE, BERTScore, PEFT, GPTQ, llama.cpp |

`full` remains the one-command compatibility profile and also contains the
`ui`, `tracking`, `documents`, and `faiss` feature stacks.

All installation profiles require Python `>=3.11,<3.13` and NumPy `>=2,<3`.
`light` does not directly install Torch, Transformers, Sentence-Transformers,
or BERTScore; those local-model dependencies are part of `full`.

`full` pins `llama-cpp-python==0.3.8`: newer source archives currently exceed
the traditional Windows path limit during pip's unpack step. The package still
normally builds its native runtime from source when installed from PyPI, so
`full` needs a C/C++ build toolchain unless you configure the project's
CPU/CUDA/Metal wheel index first. This is a native build requirement, not a
Python or NumPy resolver conflict.

On Windows, GPTQModel 7.x also pulls the native `pypcre` build. A plain
zero-toolchain `pip install "akasha-terminal[full]"` therefore still requires
Visual Studio C++ Build Tools; the upstream Windows/Python 3.12 limitation is
tracked in [GPTQModel issue #2425](https://github.com/ModelCloud/GPTQModel/issues/2425).
Use `light` when a compiler-free install is required, or prepare the native
toolchain before installing `full`.

### Editable installation for development

Use uv `0.12.13` and the committed [dependency baseline](constraints/README.md)
for the same resolved versions as CI. Create a project-local environment with
`uv venv .venv --python 3.11` before installing.

```bash
uv pip install --python .venv -e ".[light,dev]" -c constraints/light-dev.txt
```

For the complete local-model stack:

```bash
uv pip install --python .venv -e ".[full,dev]" -c constraints/full-dev.txt
```

## Configure a model provider

Set provider credentials in the environment or in a `.env` file. Never commit `.env` files or API keys.

```env
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Optional Azure OpenAI-compatible endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/

# Optional Ollama endpoint
OLLAMA_API_BASE=http://localhost:11434
```

### Gemini backends

The default Gemini Developer API configuration needs only an API key:

```env
GEMINI_API_KEY=your_key
```

To use Vertex AI **Express Mode** with an API key rather than Application
Default Credentials (ADC), use the same key and enable the explicit selector:

```env
GEMINI_API_KEY=your_key
GOOGLE_GENAI_USE_VERTEXAI=true
```

Do not set `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_LOCATION` for Express Mode.
Those variables select the full Vertex AI project/location authentication path,
which requires ADC and takes precedence over an API key. Full Vertex AI ADC
configuration is outside Akasha's API-key Express Mode support.

Supported chat model aliases include:

```text
openai:gpt-4o
gemini:gemini-2.5-flash
anthropic:claude-3-5-sonnet-latest
ollama:qwen3:8b
azure:your-deployment-name
```

Ollama can also target another host:

```text
ollama:http://192.168.1.10:11434@qwen3:8b
```

The same public interfaces accept an already configured LangChain ChatModel when provider-specific configuration is needed.

## Quick start: chat

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa("What is retrieval-augmented generation?")
print(answer)
```

`ask(stream=False)` returns a final `str`.

## Quick start: vision and image editing

Use `vision()` for image understanding. It accepts an image and a question, then returns text:

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa.vision(
    prompt="What information appears in this image?",
    image_path="input.png",
)
print(answer)
```

Use `gen_image()` to create a new image, or `edit_image()` to remove, add, or change content in an existing image:

```python
output_path = akasha.edit_image(
    prompt="Remove the bicycle and add a green potted plant",
    images="input.png",
    model="openai:gpt-image-1",
    save_path="edited.png",
)
print(output_path)
```

See the [image and vision reference](user-guide/en/reference/vision.md) for multiple-image examples.

## Quick start: RAG

RAG uses a local Chroma store and an embedding model selected independently from the chat model:

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag("./docs", "What are the main ideas in these documents?")
print(answer)
```

First-stage retrieval and second-stage reranking are configured independently:

```python
# light/full: ask the configured service-backed LLM to rank retrieved documents
rag = akasha.RAG(
    model="ollama:qwen3:8b",
    embeddings="openai:text-embedding-3-small",
    search_type="auto",
    reranker="llm",
    reranker_model="gemini:gemini-2.5-flash",  # default
    rerank_top_k=5,                             # default
)

# full only: run a BGE cross-encoder in the Akasha process
rag = akasha.RAG(
    model="ollama:qwen3:8b",
    embeddings="hf:BAAI/bge-base-en-v1.5",
    search_type="auto",
    reranker="local:BAAI/bge-reranker-base",
    rerank_top_k=5,
)
```

`rerank_top_k` keeps only the highest-ranked documents after the second-stage
reranker; it has no effect when `reranker` is not configured. For
`reranker="llm"`, `reranker_model` defaults to
`"gemini:gemini-2.5-flash"` and is created lazily, so merely constructing a RAG
instance does not require Gemini credentials. The answer model and reranker
model are configured independently. The first LLM reranking call requires
`GEMINI_API_KEY` unless `reranker_model` is overridden with another configured
provider or model object.

A custom reranker may also be a callable with the signature
`(query, documents) -> reordered_documents`. It must return every candidate
document exactly once.

Typical embedding aliases include:

```text
openai:text-embedding-3-small
gemini:gemini-embedding-001
hf:BAAI/bge-base-en-v1.5       # full installation
```

In `light`, use remote embeddings. Local HuggingFace / Sentence-Transformers embeddings require `full`.

## Quick start: agents and tools

Agents use LangChain 1.3+ native tool calling. A custom Python function can be exposed as a tool with `create_tool()`:

```python
import akasha


def today_f() -> str:
    return "The tool was called successfully."


today_tool = akasha.create_tool(
    "Return the current date or a short status message.",
    today_f,
    "today_status",
)

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[today_tool],
)

print(agent("Use the available tool and report its result."))
```

Create the agent once and reuse it for multiple questions. Rebuilding an agent for every question repeats provider initialization costs.

Set `verbose=True` to display `[progress]` (progress), `[tool]` (tool activity),
and `[answer]` (final answer). Akasha automatically adds progress instructions to
the effective system prompt while preserving your `system_prompt` and Skills.
This works with thinking enabled or disabled, including non-streaming calls.
When the model omits a progress explanation, Akasha reports the tool name without
inventing a reason or result. `agent.progress` and each saved log's `progress`
field contain the progress messages; `agent.response` contains only the answer.

## Streaming events

Non-streaming calls return a string. Streaming agents return JSON-serializable event dictionaries:

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=True,
    thinking=True,
)

for event in agent("Explain the difference between a vector store and an embedding model."):
    if event["type"] == "thinking":
        print("[thinking]", event["data"])
    elif event["type"] == "tool":
        print("[tool]", event["data"])
    elif event["type"] == "progress":
        print("[progress]", event["data"])
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

The event types are:

| Event | Meaning |
| --- | --- |
| `answer` | Final-answer text, released after the model turn is classified |
| `progress` | User-visible operation explanation, independent of thinking |
| `thinking` | Provider reasoning/thinking content, when available and enabled |
| `tool` | A tool or Skill result |

Agent text is buffered until the model turn ends, because tool calls can arrive
after text chunks. This prevents progress from leaking into `answer` events;
answer text is no longer displayed token by token. Thinking events remain
incremental. With `verbose=True`, Akasha prints the events itself, so consume the
generator without printing each event again unless you want duplicate output.

`ask(stream=True, thinking=False)` currently yields text chunks. `ask(stream=True, thinking=True)` yields `answer` and optional `thinking` events.

## Skills and MCP

Agents can load Skills from a Skill directory containing `SKILL.md`:

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=["examples/examples_skills/python-repl-skill"],
)
```

Skills can provide instructions, resources, and allowlisted tool bundles. Skill tools are surfaced through normal `tool` events.

MCP tools can be discovered with `langchain-mcp-adapters`, normalized with `akasha.normalize_mcp_tools()`, and passed to `akasha.agents(tools=...)`. The supported transports are local `stdio` and remote Streamable HTTP. New integrations should use one Streamable HTTP `/mcp` endpoint; the older HTTP+SSE transport is deprecated.

The complete example is in [`examples/ex_mcp.py`](examples/ex_mcp.py), with its server in [`examples/mcp_server.py`](examples/mcp_server.py). It uses `tool_name_prefix=True` when aggregating servers, preserves structured MCP results, and uses `stream=False` because MCP tools may be async-only.

For deterministic CI, use the local stdio fixture. Remote MCP tests must remain opt-in and should not require external credentials for the basic test suite.

## Provider loading

Provider adapters are loaded when their provider is selected:

| Model alias | Adapter |
| --- | --- |
| `openai:` / `azure:` | `langchain_openai` |
| `gemini:` | `langchain_google_genai` |
| `anthropic:` | `langchain_anthropic` |
| `ollama:` | `langchain_ollama` |

Embedding adapters follow the same rule: the relevant embedding integration is loaded only when that embedding path is used. Common LangChain core modules are still shared by all providers.

## Local development

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/iii-org/akasha.git
cd akasha
uv venv --python 3.11

# Windows PowerShell
.venv\Scripts\Activate.ps1

uv pip install --python .venv -e ".[light,dev]" -c constraints/light-dev.txt
```

Run examples:

```bash
python examples/ex_ask.py
python examples/ex_rag.py
python examples/ex_agent.py
```

## Testing

Unit tests do not require provider API keys:

```bash
python -m pytest tests -m unit
```

Image generation and editing contract tests are grouped by feature:

```bash
python -m pytest tests/vision -q
```

Focused agent and model tests:

```bash
python -m pytest \
  tests/provider/thinking/test_thinking_config.py \
  tests/agent/basic/test_core.py \
  tests/provider/factory/test_import_boundaries.py
```

Live provider tests are opt-in because they use API quota:

```powershell
$env:RUN_LIVE_TESTS = "1"
$env:ENV_FILE = "tests/.env"
python -m pytest tests/agent/stream/test_live_gemini.py -q
```

Live tests validate provider wiring, response types, tool calling, streaming events, and RAG flow. They do not evaluate the quality of model answers.

## API overview

```python
akasha.ask(...)           # document-aware QA and chat
asker.vision(...)         # image understanding and visual question answering
akasha.gen_image(...)     # image generation from a text prompt
akasha.edit_image(...)    # edit an existing image with a text prompt
akasha.agents(...)        # native tool-calling agent
akasha.RAG(...)           # document ingestion and retrieval
akasha.summary(...)       # map-reduce or refine summaries
akasha.MemoryManager(...) # persistent semantic memory
```

For detailed design decisions, upgrade notes, testing matrices, Skills, and runtime work, see ``dev_docs/``.

## License

akasha is released under the MIT License.
