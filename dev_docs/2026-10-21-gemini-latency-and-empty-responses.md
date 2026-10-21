# Gemini latency and empty response investigation

Date: 2026-10-21

## Plan

1. Reproduce with the benchmark project's Python environment (google-genai
   2.32.0, langchain-google-genai 4.4.2), explicitly loading this source checkout.
   Preserve both environments and all existing uncommitted work.
2. Compare the existing ask path, the same path with local token estimation,
   and direct synchronous LangChain. Reuse generation clients, warm each case,
   alternate order, and measure concurrency 2/4/8. Hold prompt, output limit,
   thinking settings, and transport constant. Separate count/generate timings.
3. Capture provider finish/block reasons, usage counts, visible character
   counts, content types, and actual invocation counts. Do not retain prompts,
   response text, reasoning, credentials, headers, or exception messages.
   Bound generation calls and per-call timeouts. A synthetic empty response
   demonstrates handling only; it does not establish a live upstream cause.
4. Change Gemini's default token budgeting to local estimation. Keep an explicit
   remote counting option. Preserve input budgets/document splitting; estimates
   are not provider-exact counts or billing usage.
5. Use the observed response evidence to replace blind empty-answer retries
   with an informative error and safe structured diagnostics. Keep normal text
   and thinking separation. Validate at the actual ask call boundary.
6. Re-run the benchmark and relevant offline suites; document measured effects
   and limits. No publishing, version changes, dependency upgrades, or changes
   to the benchmark project's installed package/environment.

## Predictions

- If remote counting dominates the basic gap, local estimation removes count
  requests and most time before generation, while generation time stays similar.
- If empty responses explain long tails, slow trials contain multiple generation
  attempts; provider metadata distinguishes truncation/blocking from no text.
- If the text conversion drops valid answers, the raw SDK response contains
  visible text while the normalized LangChain/Akasha answer is empty.

## Measurements

Artifacts (ignored generated data):

- `artifacts/gemini-latency/20261021-before/`: original source, 45 generation
  invocations including three warmups; default/local/direct modes.
- `artifacts/gemini-latency/20261021-after/`: modified source, 30 generation
  invocations including two warmups; default/direct modes.
- `artifacts/gemini-latency/20261021-truncation/`: eight generation invocations
  including two warmups; output ceiling deliberately reduced to one token.

All runs used `gemini-3.5-flash`, temperature 0, the same system/user content,
one warmed client per case, no SDK retries, and a 20-second per-call deadline.
Normal runs used the original 65,536 output ceiling. Dependencies came from
`C:/Users/today/Projects/akasha-gemini-test/.venv`; source came from this checkout,
not that environment's installed Akasha. Clients were constructed before timing.
The remote counter's network request was bounded in the before experiment,
but its allocation/connection behavior was otherwise unchanged.

Successful per-request mean seconds (one batch at each concurrency, no REST):

| Concurrency | Before default | Local-only substitution before edits | After default | After direct invoke |
| --- | ---: | ---: | ---: | ---: |
| 2 | 4.8079 (2/2) | 1.9758 (1/2) | 1.9493 (2/2) | 1.7988 (2/2) |
| 4 | 5.7161 (4/4) | 4.0636 (4/4) | 3.7908 (4/4) | 1.6920 (4/4) |
| 8 | 6.4794 (8/8) | 2.1537 (8/8) | 3.1812 (8/8) | 2.0856 (8/8) |

Before/after default mean reductions: 59.5%, 33.7%, 50.9%. These are small,
sequential live samples, not a guarantee of a fixed percentage. The original
table supplied by the user was REST; these numbers deliberately hold the
synchronous transport constant to isolate the token-count change.

The strongest same-run comparison is at concurrency eight: changing only the
counter reduced mean time from 6.4794 to 2.1537 seconds (66.8%). Before default
made one count request per question; after default made zero. Before default's
mean time outside generation was 3.7828 seconds, while the count endpoint itself
averaged 0.8454 seconds. The difference includes fresh client construction and
other local preprocessing; it is not all count endpoint server time. The after
run explicitly timed local token estimation: mean 0.00498 seconds including the
warmup. No import/startup time is included in these request measurements.

After-default median times were 1.9493, 2.4539, 1.9242 seconds. Three successful
requests still took 7.09–8.29 seconds: their recorded SDK generation operation
accounted for 7.09–8.28 seconds, token estimation for 0.0033–0.0104 seconds.
Removing token counting does not eliminate upstream generation variability.

## Empty response findings

Across 75 normal-ceiling calls, four upstream calls returned HTTP 429 in the
before local/direct cases; the remaining 71 returned visible text. Every trial
contained exactly one generation invocation. There were zero raw empty answers
and zero mismatches between LangChain and Akasha visible text extraction. The
old empty-response retry loop therefore did not explain this experiment's gap.
The original REST test's long tails remain unclassified because its logs did not
retain the provider evidence.

With an output ceiling of one token, seven of eight raw SDK responses contained
zero visible characters and `finish_reason=MAX_TOKENS`. This happened in both
the ask and direct LangChain cases. One ask trial instead received HTTP 504.
The three empty ask responses raised `EmptyResponseError` after one generation;
the four direct LangChain responses exposed empty text. Thus truncation can
produce empty text across packages. This deliberately constrained case does
not explain failures at the normal 65,536-token setting. No claim is made that
safety filtering, thinking-only output, or a provider bug caused the user's
original failures.

## Implementation

- Gemini budgeting now uses the existing `cl100k_base` fallback as the default
  local estimate. All existing internal callers keep their input budgets and
  document splitting. This changes estimates, so users near a context ceiling
  should allow margin; the provider remains authoritative.
- `myTokenizer.compute_tokens(..., exact=True)` explicitly requests a remote
  text count for Gemini aliases. Remote failures propagate. The helper uses a
  ten-second timeout, one SDK attempt, closes its temporary client even on
  failure, and handles Vertex Express environment selection consistently with
  chat construction. Other provider counting behavior is unchanged.
- Shared non-streaming `call_model` no longer regenerates empty responses up to
  twenty times. It raises exported `akasha.helper.EmptyResponseError` with a
  strict diagnostic allowlist. Visible partial text is returned normally.
  Tool arguments, reasoning, response text and arbitrary metadata are omitted.
- Batch/stream helpers and agent orchestration retain their existing behavior.
  This change addresses the measured single-pass ask path; it does not claim
  that every historical retry loop in the library has been removed.
- English and Traditional Chinese token-counting and ask documentation updated.

## Validation

The pre-change regression command ran the actual ask call boundary and failed
three tests: default ask sent a count request; empty output silently regenerated
and returned the next answer; the explicit remote option did not exist.

Validation results:

- New regression file: **18 passed**, covering actual ask invocation counts,
  local default/aliases/special-token strings, explicit remote counts and error
  propagation, timeout/cleanup/Vertex environment restoration, truncation,
  thinking/tool-only content, safe diagnostics, and input budget enforcement.
- Initial focused run including existing normalization/document-splitting
  checks: **17 passed** (before five additional regression cases were added).
- Expanded provider/ask/summary/observability/RAG unit run: **172 passed,
  58 deselected, one failure, seven setup errors**. All seven setup errors were
  Windows permission failures under `.pytest-tmp`; rerunning those exact tests
  under `.codex-tmp/gemini-latency-pytest-final` with the cache plugin disabled
  produced **7 passed**.
- The remaining failure is the pre-existing
  `test_english_rouge_preserves_score_without_chinese_resources`. The benchmark
  environment's installed `rouge/__init__.py` imports `rouge_chinese.rouge`.
  A standalone `import rouge` (without importing Akasha) confirms the same
  module-loading behavior. No ROUGE/environment changes were made.
- Benchmark project's existing offline tests, loading the modified source:
  **19 passed** (`unittest` discovery).
- Python compilation and `git diff --check` on owned changes passed. Original
  newline bytes on unchanged source lines were retained to avoid formatting
  churn in files with mixed historical line endings.

Pytest was run with plugin autoload disabled using the benchmark interpreter;
the repository environment's pytest installation was appended to `sys.path`
without installing or modifying either environment. Live tests were not enabled
in the offline suite. New API calls are limited to the three explicitly live
experiments above (83 total generation invocations).

Live reproduction (PowerShell, choose a new output directory each time):

```powershell
$benchPython = 'C:/Users/today/Projects/akasha-gemini-test/.venv/Scripts/python.exe'
$benchEnv = 'C:/Users/today/Projects/akasha-gemini-test/src/akasha_gemini_test/.env'
& $benchPython scripts/benchmark_gemini_latency.py --live --env-file $benchEnv `
  --output artifacts/gemini-latency/new-run --modes default direct `
  --concurrency 2 4 8 --max-generations 36 --timeout 20

& $benchPython scripts/benchmark_gemini_latency.py --live --env-file $benchEnv `
  --output artifacts/gemini-latency/new-truncation --modes default direct `
  --concurrency 1 --repeats 3 --max-generations 8 --max-output-tokens 1 --timeout 20
```

The env file is selected explicitly; use the project-root `.env` instead if
that is where the benchmark credentials are stored. The script never installs
or edits packages, dependencies, credentials, or benchmark-project code.

## Installed benchmark environment update

At the user's subsequent request, built the current checkout into
`artifacts/gemini-latency/installed-update-20261021/dist/akasha_terminal-1.8.3-py3-none-any.whl`
and installed that wheel into `akasha-gemini-test/.venv` with
`uv pip install --no-deps --reinstall-package akasha-terminal`. This is a local
build with the existing 1.8.3 version; no release or dependency upgrade occurred.

Fresh-process verification from the benchmark project (no source-path injection):

- Loaded `akasha-gemini-test/.venv/Lib/site-packages/akasha`.
- Installed changed source files matched the checkout byte for byte.
- All installed package versions matched the pre-install snapshot.
- `pyproject.toml` and `uv.lock` hashes were unchanged.
- Default Gemini counting created zero network clients; an empty ask response
  raised `EmptyResponseError` after exactly one generation.
- Existing benchmark suite: **19 passed** against the installed package.
- `uv pip check`: **158 packages checked, all compatible**.

Snapshots and provenance are saved under the same `installed-update-20261021`
artifact directory (`before.json`, `verification.json`). A read-only
`uv sync --dry-run --locked` confirmed that syncing would replace this local
wheel with registry 1.8.3. For this local test build, invoke the venv Python
directly or use `uv run --no-sync ...`; project dependency configuration was
deliberately preserved.

## v1.8.4 release validation

PR #30 includes the pending feature-loading changes and the Gemini changes.
The repository already had secrets named `GOOGLE_GENAI_USE_VERTEXAI` and
`OLLAMA_API_BASE`, but CI did not forward them. The release adds that wiring;
`PYPI_API_TOKEN` also exists (only names were inspected, not secret values).

The first PR CI attempt passed 184 checks before the Gemini live smoke test
received an empty normalized answer with `finish_reason=STOP`, 57 input tokens,
334 output tokens, and 391 total tokens. Vertex mode was enabled. This is a
normal-ceiling empty answer, distinct from the controlled one-token truncation
experiment. The available log does not establish whether the SDK response was
already empty or its conversion lost text. The live smoke test now records
raw SDK character counts, part counts, finish reasons, and usage to distinguish
those cases, without logging answer or reasoning text. No library retry was
reintroduced. A subsequent commit starts one bounded new CI attempt because
the available API credentials cannot invoke the Actions rerun endpoint.

Local release checks: wheel/sdist build and `twine check` passed; wheel/sdist
boundaries passed; dependency baseline exports match the lock after changing
only the project version to 1.8.4. The broad offline run passed 329 tests with
one skipped; one caller-runtime test could not find pytest in its child process
because the test interpreter borrowed pytest through the parent process's
`sys.path`. That case is rerun with the same pytest path inherited by the child.
