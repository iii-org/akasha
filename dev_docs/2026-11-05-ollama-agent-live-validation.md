# Ollama Agent live validation — 2026-11-05

## Scope

- Endpoint: a user-provided temporary Cloudflare Tunnel (omitted from this public report).
- Model: `hf.co/nctu6/Gemma-3-TAIDE-12b-Chat-GGUF:Q4_K_M`
- Server `/api/version`: `0.40.14`; `/api/show` capabilities: `completion`.
- Entry point: the public `akasha.agents` interface, including real local tool
  execution and real model responses. No fake model responses were used in the
  live cases. Verification codes were generated during each test, outside the
  model prompt.
- The original official structured-chat integration was retained. Experimental
  completion reminders and structured-observation prompt changes tried during
  diagnosis were withdrawn because they did not improve acceptance reliably.

## Results

The latest result for each of nine cases is **3 passed, 6 failed**, collected in
two groups against the retained integration. This is not a fully passing live
acceptance result.

| Case | Result | Evidence |
| --- | --- | --- |
| Tool lookup, synchronous call | Fail | Instrumented function executed twice for record 42 rather than once. |
| Tool lookup, `await agent.acall(...)` | Fail | The same duplicate execution. |
| Tool lookup, streaming call | Fail | The same duplicate execution. |
| Date → add 77 → shuffle digits, non-streaming | Fail on rerun | All three functions ran with correct arguments, but the answer quoted `[2, 1, 0, 3]` instead of the actual shuffle result `[2, 1, 3, 0]`. |
| Date → add 77 → shuffle digits, streaming | Pass | Three tools executed in order, arguments depended on actual prior results, and the answer quoted the actual shuffled list. |
| Plain `ask` and no-tool Agent, non-streaming | Pass | Both returned the requested visible token. |
| Plain `ask` and no-tool Agent, streaming | Pass | Both returned the requested visible token and the Agent recorded no tool calls. |
| Dynamic Skill and its tool | Fail | The Skill's function executed twice for record 42. |
| Same Agent called twice | Fail | An answer invented a verification string rather than quoting the actual function result. This failed before the complete two-call acceptance could be established. |

The first five-case run passed both three-tool workflows, but the later
non-streaming rerun failed result fidelity. Those earlier successes therefore
do not establish stable correctness. The three lookup cases failed the
exactly-once check in the first run; the other six cases were subsequently run
against the retained version as a separate group.

During a withdrawn prompt experiment, capturing the input to LangChain's
`JSONAgentOutputParser` also showed a fabricated verification value in the
model's original `Final Answer` JSON. The observed errors include model
decisions and answer fidelity under these prompts, not only connectivity or
whether the Python functions execute.

## Local checks

- Broad offline regression during this investigation: **351 passed, 1 skipped**;
  live and full-profile tests were excluded.
- Retained integration's focused offline contracts: **10 passed**, covering
  sync/async/stream calls, dynamic Skills, hidden runtime arguments, validation,
  round limits, history, thinking/usage metadata, and invalid JSON propagation.
- These offline successes do not replace the failing live acceptance above.

## Reproduce

```powershell
$env:RUN_LIVE_TESTS = "1"
$env:AKASHA_OLLAMA_AGENT_MODEL = "ollama:https://your-server@hf.co/nctu6/Gemma-3-TAIDE-12b-Chat-GGUF:Q4_K_M"
.venv/Scripts/python.exe -m pytest tests/agent/basic/test_live_ollama_json.py -q --tb=short
```

The URL is an ephemeral tunnel and may need replacement. Raw run logs remain
locally under `.codex-tmp/`, including `ollama-new-endpoint-live.log` (first five
cases), `ollama-new-endpoint-retained-version.log` (six remaining cases), and
separately named diagnostic runs. Failures were not converted to skips or
silently retried within tests.
