# Examples refresh validation (2026-09-18)

## Scope and behavior

Updated all 22 executable Python examples under `examples/`, including helper,
Skills, API client and MCP server examples. Examples now have import-safe main
entrypoints, `--help`, current public API calls and consistent configuration.
Bundled synthetic documents replace missing PDFs and private directories.

Shared setup merges dotenv into the process environment without overriding
existing variables, supports CLI model/embedding selection, and saves outputs in
ignored, isolated directories. Short, unique relative document paths avoid
Windows Chroma path length and shared-client cache collisions. The deletion
example works only on copies in its private index. Memory uses a stable directory.
Agent and Skills examples consume streams without duplicate verbose printing.

Necessary library fixes found by executing the examples:

- `update_db()` no longer stops Chroma's shared cached system after metadata
  updates; a subsequent client can reload the collection.
- Self-ask resolves intermediate questions into strings before final synthesis,
  restores the requested stream mode, and records the completed streamed answer.
- RAG starts each streamed answer with an empty response accumulator.

The previously existing agent-progress changes and other worktree changes were
preserved. No commit was created.

## Automated validation

Interpreter: `..\.venv\Scripts\python.exe` (existing Windows environment).

- Full local selection: **256 passed, 62 deselected, 4 warnings**, exit code 0.
  Command: `python -m pytest -m "not live and not full_only" -o addopts='' -q`.
- Examples alone: **57 passed, 2 warnings**, exit code 0.
  Command: `python -m pytest tests/examples -o addopts='' -q`.
- Both runs used a short unique `--basetemp` under the system temporary directory
  to keep real Chroma integration paths within Windows limits.
- Compilation and Git whitespace checks passed for the changed implementation.

Coverage includes all CLI entrypoints from an unrelated working directory, import
safety, public signatures, stream rendering, API payload schema, real Chroma
load/delete/retrieval/metadata behavior, the actual MCP stdio server, configured
search engine selection, and Self-ask streams with and without follow-up questions.
The metadata and Self-ask regressions were observed failing before their fixes.
External model/embedding calls in these local tests use deterministic doubles.

Warnings include LangChain deprecations. During the full run Windows also emitted
an access-violation diagnostic while importing optional pyarrow/pandas/transformers
code; pytest continued and exited 0 with the totals above. This is an environment
caveat, not evidence that those optional native dependencies are healthy.

## Actual service execution

Live checks used the existing `tests/.env`, default OpenAI chat/embedding models,
synthetic example data and isolated output directories. These are execution smoke
checks, not assessments of model answer quality or every supported provider.

Successful executions:

- Ask: ordinary, streaming and vision using the generated local image.
- RAG and Self-ask: ordinary and streaming (Self-ask rerun after its fix).
- Summary: map-reduce and refine.
- Agent: ordinary and streaming with inventory/delivery tools.
- Skills: ordinary, streaming and persistent Python REPL examples.
- Helpers: model objects, load/reload DB, extraction/deletion, retrieval,
  metadata/self-query (rerun after fix), token counting, local ROUGE and LLM scoring.
- Persistent long-term memory.
- Evaluation question generation: general and topic-specific.
- Brave web search, both explicit selection and automatic key-based selection.
  The initial Wikipedia request failed with a remote connection
  closure; the example now supports explicit selection and automatic selection
  from available search keys.
- Actual local Streamable HTTP MCP server/client; API ask, summary and RAG requests
  against a temporary local Akasha HTTP server. Servers were stopped afterward.
- Image generation and image editing with `openai:gpt-image-1`, both producing PNGs. PIL verified both 1024 x 1024 image files.

Local JSON summaries and redacted stdout/stderr logs are in
`examples/output/validation/` (ignored by Git). Earlier failed-case records are
retained alongside successful reruns to preserve the diagnostic history.

## Limits

BERT model downloads and full `--evaluate` scoring were not executed. API websearch,
all alternative providers/search engines, every thinking-capable model, remote
server deployment and a fresh package installation were not exhaustively tested.
Provider credentials, account access, network availability and optional full
scoring dependencies remain prerequisites; see `examples/README.md`.
