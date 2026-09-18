# Functional Test Coverage Specification

Status: implemented and reorganized in September 2026

## Purpose

The suite must make three facts independently visible:

1. which product feature owns a test;
2. whether it is deterministic, local integration, or external-service work;
3. which optional contract or dependency profile it verifies.

A green local unit run must not be presented as provider, packaging, or
end-to-end proof. Current run evidence and encountered failures are maintained
in `dev_docs/2026-09-test-suite-reorganization-worklog.md`.

## Implemented structure

Tests use a feature-first layout:

```text
tests/
  agent/          ask/            cli/            db/
  eval/           mcp/            memory/         observability/
  provider/       rag/            skill/          summary/
  tool/           vision/
  support/        data/           fixtures/       config/
```

Every module is named `test_*.py`. Tests formerly located below
`tests/utils/` or broad cross-feature compatibility modules now live with the
feature they exercise. Old paths are deliberately not retained.

Shared runtime helpers live in `tests/support/`:

- `fakes.py` owns reusable fake chat models;
- `live.py` owns the single live-test gate, credential checks, and Ollama
  reachability probe;
- `paths.py` owns repository fixture paths.

The canonical large PDF corpus is `docs/mic/`. Byte-identical PDF copies were
removed from `tests/data/documents/`; only test-specific structured and DOCX
fixtures remain there.

## Marker contract

Every collected test has exactly one execution tier:

| Tier | Meaning |
|---|---|
| `unit` | Deterministic, local, and free of network or real-service calls |
| `integration` | Real local components, processes, HTTP, or MCP without an external service |
| `live` | External network or real provider/service access |

`tests/conftest.py` rejects tests with zero or multiple tier markers.
Feature markers are derived from the first directory below `tests/`, keeping
feature filtering without repeated declarations.

Orthogonal markers are `contract`, `smoke`, `full_only`, `requires_api`,
and `upgrade`. In particular, `contract` is not an execution tier.

## Live-service contract

`RUN_LIVE_TESTS=1` is the only switch that enables external tests. Each
provider case checks only its own required configuration and skips
independently when that configuration is absent. A configured provider that
returns an API or contract error fails normally.

Ollama is eligible only when `OLLAMA_API_BASE` is explicitly configured and
its `/api/version` endpoint responds. Locally started MCP and HTTP fixture
servers remain `integration`, not `live`.

The provider RAG suite is one parameterized contract:

- OpenAI chat plus OpenAI embeddings;
- Gemini chat plus Gemini embeddings;
- Anthropic chat plus OpenAI embeddings.

All cases must retrieve the stable fixture signal `RAG-7319-TAIPEI`, include
that exact signal in the answer, retain supporting documents, and expose
JSON-serializable logs. Gemini is also the sole detailed staged RAG pipeline,
covering embedding, Chroma persistence/reload, retrieval, and final generation.

## Full dependency contract

The light profile may deliberately omit local-model and evaluation packages.
The full profile may not: missing `accelerate`, `bert_score`, `gptqmodel`,
`torch`, or `transformers` is a packaging failure.

The GPTQ test is both `live` and `full_only`. It loads
`ModelCloud/Qwen2.5-0.5B-Instruct-gptqmodel-w4a16` at immutable revision
`7911278f62450ff588b7f1bb6c0778fba839ede8` on CPU and completes one
inference. It must not silently substitute a mock backend or require CUDA.

## Coverage ownership

| Capability | Primary evidence | Tier / contract | Status |
|---|---|---|---|
| Ask prompt and info normalization | `tests/ask/prompt/`, `tests/ask/info/` | unit | covered |
| Ask real provider and URL behavior | `tests/ask/basic/test_live_ask.py`, `tests/ask/info/url/` | live, smoke | covered by configured environment |
| Agent public output and event schema | `tests/agent/contracts/`, `tests/agent/stream/` | unit contract + live | covered |
| Database build/extract/delete and Path inputs | `tests/db/basic/`, `tests/rag/input/test_db_structure.py` | live smoke + unit | covered |
| Evaluation | `tests/eval/basic/test_eval.py` | live, full_only | covered by full environment |
| MCP stdio and Streamable HTTP | `tests/mcp/` | integration contract + live agent bridge | covered |
| Memory retrieval, persistence, and provider flow | `tests/memory/` | unit + live smoke | covered |
| Observability and serializable logs | `tests/observability/` and feature assertions | unit contract | covered |
| Chat provider matrix and thinking | `tests/provider/chat/test_provider_contract.py` | live contract smoke | covered by provider/service |
| Embedding provider matrix | `tests/provider/embedding/` | live contract | covered by provider |
| Optional dependency and packaging behavior | `tests/provider/compatibility/` | unit/full contract | covered |
| RAG parameters and retrieval | `tests/rag/parameters/`, `tests/rag/retrieval/` | unit contract | covered |
| RAG provider matrix and staged Gemini flow | `tests/rag/provider/` | live contract smoke | covered by provider |
| Skill loading, resources, scripts, and safety | `tests/skill/` | unit contract | covered |
| Summary | `tests/summary/basic/test_summary.py` | live smoke | covered by provider |
| Tool schema | `tests/tool/schema/` | unit | covered |
| Vision adapters and generated-image flow | `tests/vision/` | unit + live smoke | covered by provider |

## CI contract

The main matrix installs `.[light,dev]` on Python 3.11 and 3.12, enables the
single live switch, and runs `not full_only`. GitHub secrets determine which
provider cases are eligible; no separate per-provider run switch exists.

A Python 3.11 full job installs `.[full,dev]`, caches Hugging Face model
artifacts, and runs `full_only`. Dependency resolution in the light job checks
that the full profile remains solvable without making the light job install it.

## Acceptance criteria

- Collection succeeds and marker policy accepts every test.
- No collected test function consists only of `pass`.
- No `*_test.py` module remains.
- `pytest -m "not live and not full_only"` passes.
- All eligible `live and not full_only` cases pass; unavailable Ollama or
  missing credentials are explicit per-case skips.
- `full_only` passes in a full Linux CPU environment, including real GPTQ
  inference.
- `pytest-cov` is the only coverage runner.
- Repository-wide unit coverage meets the explicit 48% baseline; the previous
  custom runner's hand-picked subset is not reported as whole-project coverage.
- `.pytest-tmp/` remains ignored and available for inspection.
- `tests/README.md`, `tests/TESTING_GUIDE.md`, `pytest.ini`, and CI all
  describe the same commands and marker semantics.

## Non-goals

- Exhaustively multiplying every provider by every RAG parameter.
- Treating model wording as deterministic when a fixture-specific exact signal
  is sufficient.
- Keeping compatibility aliases for obsolete test paths.
- Calling a local fixture server `live`.
- Hiding configured-provider failures as skips.
