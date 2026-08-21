# Akasha

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)

Akasha is a Python toolkit for document question answering, retrieval-augmented generation (RAG), native tool-calling agents, summaries, and long-term semantic memory.

It provides one consistent interface for remote and local model workflows while keeping provider-specific integrations behind model aliases such as `openai:`, `gemini:`, `anthropic:`, and `ollama:`.

- Chinese manual: <https://iii-org.github.io/akasha/>
- Current package version: `1.5`

## What Akasha provides

| Capability | Public entry point | Purpose |
| --- | --- | --- |
| Chat / QA | `akasha.ask()` | Ask a model a question, optionally with documents or web information |
| Agents | `akasha.agents()` | Use LangChain-native tool calling, streaming, thinking events, Skills, and MCP tools |
| RAG | `akasha.RAG()` | Load documents, create embeddings, search Chroma, and generate an answer |
| Summaries | `akasha.summary()` | Summarize text, files, or URLs with `map_reduce` or `refine` |
| Long-term memory | `MemoryManager` | Store and retrieve semantic memories with Chroma |

## Installation

Python 3.11 or 3.12 is recommended.

### Lightweight installation

Use `light` for remote chat models and remote embeddings:

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

`light` keeps Chroma-backed RAG and memory workflows, but does not include the local HuggingFace / Torch model stack.

### Full installation

Use `full` when you need local embeddings, local HuggingFace models, local Llama/GPTQ models, reranking, or BERTScore:

```bash
uv pip install "akasha-terminal[full]"
```

The practical difference is:

| Installation | Chat models | Embeddings | Vector store | Local ML / rerank |
| --- | --- | --- | --- | --- |
| `light` | Remote providers | Remote APIs | Local Chroma | No |
| `full` | Remote and local providers | Remote and local | Local Chroma | Yes |

`light` does not include Torch, Transformers, Sentence-Transformers, or `onnxruntime`; these local-model dependencies are part of `full`.

### Editable installation for development

```bash
uv pip install -e ".[light,dev]"
```

For the complete local-model stack:

```bash
uv pip install -e ".[full,dev]"
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
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

The event types are:

| Event | Meaning |
| --- | --- |
| `answer` | A chunk of the final answer |
| `thinking` | Provider reasoning/thinking content, when available and enabled |
| `tool` | A tool or Skill result |

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

uv pip install -e ".[light,dev]"
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

Focused agent and model tests:

```bash
python -m pytest \
  tests/provider/thinking/test_thinking_config.py \
  tests/agent/basic/test_core.py \
  tests/provider/factory/test_import_boundaries.py
```

Live provider tests are opt-in because they use API quota:

```powershell
$env:RUN_LLM_TESTS = "1"
$env:ENV_FILE = "tests/.env"
python -m pytest tests/agent/stream/test_live_gemini.py -q
```

Live tests validate provider wiring, response types, tool calling, streaming events, and RAG flow. They do not evaluate the quality of model answers.

## API overview

```python
akasha.ask(...)           # document-aware QA and chat
akasha.agents(...)        # native tool-calling agent
akasha.RAG(...)           # document ingestion and retrieval
akasha.summary(...)       # map-reduce or refine summaries
akasha.MemoryManager(...) # persistent semantic memory
```

For detailed design decisions, upgrade notes, testing matrices, Skills, and runtime work, see ``dev_docs/``.

## License

Akasha is released under the MIT License.
