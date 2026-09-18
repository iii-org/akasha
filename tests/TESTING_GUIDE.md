# Akasha Testing Guide

This is the canonical guide for the Akasha test suite. The short
[README.md](README.md) is only a command entry point.

## Environment

Use the project-local environment installed from the committed dependency baseline:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[light,dev]" -c constraints/light-dev.txt
$python = ".venv\Scripts\python.exe"
```

See [constraints/README.md](../constraints/README.md) for full mode and controlled
updates. The older parent-directory environment is not the CI baseline. Pytest
sets `pythonpath = .`, so local deterministic tests exercise the current source
tree. Packaging contract tests separately verify installed dependency profiles.

## Layout

Tests are owned by the feature they exercise:

| Feature | Directory | Representative coverage |
|---|---|---|
| Agent | `tests/agent/` | public API, streaming, final actions |
| Ask | `tests/ask/` | prompts, local parsing, URLs, live Ask |
| CLI | `tests/cli/` | optional UI boundary |
| Database | `tests/db/` | build, load, extract, delete |
| Evaluation | `tests/eval/` | model evaluation with full dependencies |
| MCP | `tests/mcp/` | stdio, Streamable HTTP, result contracts |
| Memory | `tests/memory/` | deterministic retrieval and live persistence |
| Observability | `tests/observability/` | console, files, JSON-safe logs |
| Providers | `tests/provider/` | chat, embeddings, compatibility, thinking |
| RAG | `tests/rag/` | input, parameters, retrieval, provider matrix |
| Skills | `tests/skill/` | loading, resources, scripts, safety |
| Summary | `tests/summary/` | live summarization |
| Tools | `tests/tool/` | tool schema |
| Vision | `tests/vision/` | deterministic adapters and live image flow |

Reusable fakes and environment helpers live in `tests/support/`. Stable
fixtures live in `tests/data/`. The canonical PDF corpus lives in
`docs/mic/`; tests must not copy those PDFs into `tests/data/documents/`.

All collected test modules use the `test_*.py` naming convention. Old paths
are not retained as compatibility aliases.

## Marker policy

Every collected test must have exactly one execution tier:

- `unit`: deterministic and local; no network or real service.
- `integration`: combines real local components, processes, HTTP servers, or
  MCP servers without contacting an external service.
- `live`: contacts an external network or a real provider/service.

`tests/conftest.py` rejects missing or duplicate execution tiers during
collection. It also derives one feature marker from the first directory below
`tests/`.

The following markers are orthogonal and may accompany a tier:

- `contract`: public API, schema, serialization, packaging, or compatibility.
- `smoke`: minimal end-to-end proof of a critical path.
- `full_only`: requires the `full` dependency profile.
- `requires_api`: requires provider credentials.
- `upgrade`: dependency-upgrade regression.

Example:

```python
pytestmark = [
    pytest.mark.live,
    pytest.mark.contract,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]
```

## Common commands

```powershell
$python = ".venv\Scripts\python.exe"

# Collection and naming/marker validation
& $python -m pytest --collect-only -q

# Execution tiers
& $python -m pytest -m unit -q
& $python -m pytest -m integration -q
& $python -m pytest -m "not live and not full_only" -q

# Feature selections
& $python -m pytest tests/ask -q
& $python -m pytest tests/memory -q
& $python -m pytest tests/mcp -q
& $python -m pytest -m rag -q

# Coverage; pytest-cov is the only coverage runner
& $python -m pytest -m unit --cov --cov-report=term-missing
```

The repository-wide unit baseline is 48% (48.72% when this layout was
introduced). The previous custom script measured only a hand-picked subset and
was not comparable to repository-wide coverage. Raise the configured floor as
new deterministic tests are added.

Pytest writes temporary artifacts below `.pytest-tmp/`. The directory is
ignored but intentionally retained after a run for failure inspection.

## Live tests

All external tests share one opt-in:

```powershell
$env:RUN_LIVE_TESTS = "1"
& $python -m pytest -m live -q
```

Credentials are read from process environment variables first, then from the
file selected by `ENV_FILE`, or `tests/.env` when no explicit file is set.
Secret values must never be printed.

Provider cases are independent:

- OpenAI uses `OPENAI_API_KEY`.
- Gemini uses `GEMINI_API_KEY`.
- Anthropic uses `ANTHROPIC_API_KEY`; its RAG case also needs
  `OPENAI_API_KEY` for embeddings.
- Azure uses `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_BASE_URL`.
- Brave-backed web tests use `BRAVE_API_KEY`.
- Ollama runs only when `OLLAMA_API_BASE` is explicitly set and
  `/api/version` is reachable.

A missing key skips only the affected provider. Once a provider is configured,
its API failure is a real failure, not a skip. Live tests may incur charges.

## Full-only tests

A full installation treats missing full-profile packages as a packaging error:

```powershell
uv pip install -e ".[full,dev]" -c constraints/full-dev.txt --python $python
& $python -m pytest -m full_only -q
```

The GPTQ contract downloads
`ModelCloud/Qwen2.5-0.5B-Instruct-gptqmodel-w4a16` at revision
`7911278f62450ff588b7f1bb6c0778fba839ede8`, loads it through GPTQModel on
CPU, and performs one inference. CI caches Hugging Face downloads. A light
profile may intentionally omit these packages; a full profile may not.

## CI

The light matrix runs Python 3.11 and 3.12 with `.[light,dev]`, sets
`RUN_LIVE_TESTS=1`, and runs every test except `full_only`. Available
provider secrets run automatically; absent secrets skip only their cases.

The Python 3.11 full job installs the native compiler and Git required by
source-only dependencies, installs `.[full,dev]`, restores the Hugging Face
cache, and runs `full_only`. Ollama is not assumed to exist in CI.

## Writing and reviewing tests

- Assert observable results: return values, normalized events, persisted state,
  selected documents, tool calls, errors, files in temporary roots, and
  JSON-serializable logs.
- Do not leave placeholder tests whose body is only `pass`.
- Use `tmp_path` or another dedicated temporary root for writable state.
- Prefer shared helpers from `tests/support/` over cross-importing another
  feature's test module.
- Keep exact fixture-specific signals in live assertions. A merely non-empty
  model response does not prove grounded behavior.
- Classify a failure before changing it: fix the test when setup or expectation
  is wrong; fix product code when the documented public behavior is wrong.

The reorganization decisions and test evidence are recorded in
`dev_docs/2026-09-test-suite-reorganization-worklog.md`.
