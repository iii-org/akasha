# Akasha examples

These examples use the current public API. Importing a module does not call a
model, create an index, or start a server. Run a file directly or use
`python -m examples.ex_ask`; paths work even when the current directory is
outside this checkout.

## Setup

Install the project in your environment:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv -e ".[light,dev]" -c constraints/light-dev.txt
```

For this checkout, the baseline interpreter is
`.venv\Scripts\python.exe`. Activate it with `.\.venv\Scripts\Activate.ps1`
before using the `python` commands below.

Put credentials in the repository-root `.env` (see [.env.example](.env.example)),
set them in the process environment, or pass `--env-file PATH`.
The precedence is: explicit CLI model/embedding options, environment variables,
then defaults. Existing environment variables override values from the dotenv
file. Examples never require a private `.env3` or a Skills-local `.env`.

Defaults:

- Chat: `AKASHA_MODEL=openai:gpt-4o-mini`
- Embeddings: `AKASHA_EMBEDDINGS=openai:text-embedding-3-small`
- Images: `AKASHA_IMAGE_MODEL=openai:gpt-image-1` (image API access required)
- Chat and embedding models can use different providers; configure both keys.

For example:

```powershell
python examples/ex_ask.py --help
python examples/ex_ask.py
python examples/ex_rag.py --stream
python examples/ex_agent.py --stream
python examples/ex_ask.py --model gemini:gemini-2.5-flash
python examples/ex_rag.py --model gemini:gemini-2.5-flash --embeddings gemini:gemini-embedding-001
```

## Examples and prerequisites

| Entry point | Demonstrates | Requires |
| --- | --- | --- |
| `ex_ask.py` | Local reference text; `--stream`; optional `--image PATH_OR_URL` | Chat key; vision-capable model for images |
| `ex_rag.py` | Retrieval and source references; `--stream` | Chat + embedding keys |
| `ex_selfask_rag.py` | Question decomposition and RAG; `--stream` is configured on the instance | Chat + embedding keys |
| `ex_summary.py` | `--method map_reduce` or `refine` | Chat key |
| `ex_websearch.py` | `--engine wiki`, `serper`, `brave`, or `tavily`; optional `--stream` | Chat key + network; search key except wiki |
| `ex_agent.py` | Local inventory/delivery tools and progress; `--stream`, `--thinking` | Chat key; thinking-capable model if enabled |
| `ex_mcp.py` | Streamable HTTP tools via `await agent.acall()` | Chat key + running MCP server |
| `mcp_server.py` | Deterministic add and demo-weather tools | Local MCP dependencies; no model key |
| `ex_long_term_memory.py` | Persist and retrieve conversation memory | Chat + embedding keys |
| `ex_eval.py` | Generate a question set; optional `--topic` and `--evaluate` | Chat + embedding keys; full scoring dependencies for evaluation |
| `ex_generate_img.py` | Generate an image; opt into a second call with `--edit` | Image-provider key and model access |
| `ex_api.py` | `--action ask/rag/summary/websearch` | Running Akasha API + relevant provider keys |
| `helper/ex_handle_obj.py` | Model calls, batching, streaming, JSON formatting, embeddings | Chat + embedding keys |
| `helper/ex_load_db.py` | Build/reload Chroma and inspect documents | Embedding key |
| `helper/ex_remove_db.py` | Extract/pop chunks and delete stored documents from a private demo index | Embedding key |
| `helper/ex_retriver.py` | Ranked retrieval, scores, `search_docs` and `retri_docs` | Embedding key |
| `helper/ex_self_query.py` | Add/persist metadata and filter factory reports | Chat + embedding keys |
| `helper/ex_scores.py` | Local ROUGE; optional `--llm` and `--bert` | None by default; chat key for LLM; full dependencies/model download for BERT |
| `helper/ex_token_count.py` | Token/word counts, Chinese conversion, JSON extraction | No model API; tokenizer may download its encoding on first use |
| `examples_skills/app_non_stream.py` | Load and execute the greeting Skill | Chat key |
| `examples_skills/app_stream.py` | Same Skill with streamed events | Chat key |
| `examples_skills/repl_app.py` | Reuse Python variables between tool calls | Chat key |

The small documents under `data/` are synthetic example inputs. They replace
missing files such as `docs/1.pdf` and private report directories. No PDF corpus
or remote GitHub page is necessary for the basic Ask/RAG examples.

Search engine selection is `--engine`, then `AKASHA_SEARCH_ENGINE`, then an
available Brave/Serper/Tavily key, falling back to Wikipedia. Wikipedia requires
network access and may be blocked in some environments. To choose explicitly:
`python examples/ex_websearch.py --engine brave`.

## MCP and HTTP API

In separate terminals:

```powershell
python examples/mcp_server.py
python examples/ex_mcp.py
```

MCP defaults to `http://127.0.0.1:8001/mcp`; override the server with
`--port`/`MCP_PORT` and the client with `--url`/`MCP_URL`.
The old `AKASHA_MCP_MODEL` override is still supported.
MCP tools are async-only; this example uses non-streaming `acall()`.
`mcp_server.py --transport stdio` is also available for local integration.

The Akasha API defaults to port 8000:

```powershell
python -m uvicorn akasha.api:app --host 127.0.0.1 --port 8000
python examples/ex_api.py --action ask
python examples/ex_api.py --action summary
```

The client sends provider configuration from your environment in the API's
`env_config` field; use your own trusted Akasha server. It does not send
placeholder keys. Set `--base-url` or `API_BASE_URL` to change the endpoint.
For remote RAG, `--data-source` must be a path visible to the **server**.

## Output and streaming

Generated artifacts are saved under `examples/output/` (ignored by Git).
Use `--output-dir` to change it. Most runs get a new isolated subdirectory,
which is printed at the end. Chroma caches use short relative source paths to
avoid Windows path-length issues. The deletion example only indexes copies of
bundled documents in its own run directory; it does not delete source files or
your existing indexes.

Long-term memory uses a stable `output/memory/` directory so it can survive
subsequent runs. Change `--memory-name` for another collection.

Agent/Skills examples use `verbose=True`. They consume stream events without
printing them a second time. `[progress]`, `[tool]`, and `[answer]` work
independently of thinking. Agent answer text is emitted after a model turn is
classified; it is not token-by-token text. Ask/RAG streams are consumed by
`print_response()`, which handles both string chunks and event dictionaries.

Image generation, image editing, BERT downloads and evaluation are explicit
operations; invoking `--help` does none of them.

## Verification

`tests/examples/test_examples.py` checks CLI entrypoints from an unrelated
directory, import safety, public call signatures, stream rendering, API payloads
and real local Chroma workflows with the embedding API replaced by a deterministic
test double. Live provider execution is recorded separately; local tests do not
establish model quality or account/model availability.

See the [execution report](../dev_docs/2026-09-examples-refresh.md) for local test
results, actual provider/service checks, and unverified optional paths.
