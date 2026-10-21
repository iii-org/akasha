# Same Gemini Adapter: Direct LangChain vs Akasha

Date: 2026-10-09. Existing light test environment and current modified source.
No production changes, installation changes, or credential changes.

## Fairness Controls

Used `scripts/benchmark_adapter_vs_akasha.py`, three sequential rotated rounds.
Each case ran first with a unique empty bytecode prefix, then in another new
process with its own preceding prefix reused. No imports were reused in memory.
Bytecode writing was enabled throughout. No import tracing, cProfile, or ETW.

- Same Python 3.11.11, Gemini 2.5 Flash, API key, temperature=0, output limit=128.
- Same base adapter: `langchain_google_genai.ChatGoogleGenerativeAI` 4.3.3.
- LangChain Core 1.6.3; Google GenAI SDK 2.18.0.
- Direct case did not import any Akasha modules.
- Direct case used a tiny local subclass replicating ONLY Akasha's existing
  `_prepare_request` override to disable SDK automatic function calling.
- Explicit identical system instruction and human question. Akasha used
  `prompt_format_type="chat_gpt"` so both sent system/human messages; direct
  input included Akasha's `User question:` prefix. This tests the full public
  ask path, with Chinese language default, no documents, history, or tools.
- Neither side explicitly set a native thinking budget. Akasha thinking=False.
- Parent supplied credentials through the environment; env_file="" on Akasha.
- Disabled service tracing and ambient Vertex/ADC settings on both sides.

Hashed sanitized prepared GENERATION request fields: model, contents, system
instruction, temperature, token limit, thinking config, automatic function
calling config, and tools. All twelve hashes were identical. This does not
mean the total HTTP request counts match: Akasha also counts input tokens.
Hashing and a lightweight invoke timer were the only call instrumentation.
No raw credentials or response bodies were captured.

## Results

All twelve real generation requests returned exactly ASK_SMOKE_OK. Every
answer used one adapter invoke and one prepared generation request.

| First complete answer | Direct LangChain | Akasha |
| --- | ---: | ---: |
| Cold round 1 | 18.208 s | 18.486 s |
| Cold round 2 | 16.928 s | 27.563 s |
| Cold round 3 | 17.183 s | 16.515 s |
| Cold median | 17.183 s | 18.486 s |
| Warm round 1 | 5.653 s | 5.901 s |
| Warm round 2 | 3.654 s | 7.423 s |
| Warm round 3 | 3.578 s | 6.061 s |
| Warm median | 3.654 s | 6.061 s |

Differences between medians: cold +1.303 s; warm +2.407 s. These are observed
sample differences, not fixed Akasha overhead estimates. Cold paired deltas
range from -0.668 s to +10.636 s. I/O and network vary, and three rounds are
not enough to turn the cold delta into a causal constant.

| Phase, independent median | Direct cold | Akasha cold | Direct warm | Akasha warm |
| --- | ---: | ---: | ---: | ---: |
| Ready: import + construction | 15.869 s | 14.670 s | 2.294 s | 2.310 s |
| First call | 1.384 s | 3.816 s | 1.360 s | 3.690 s |
| Inside adapter invoke | 1.384 s | 1.440 s | 1.360 s | 1.375 s |
| Outside adapter invoke | ~0 s | 2.376 s | ~0 s | 2.309 s |

Independent medians do not necessarily sum to the median total. The invoke
boundary includes adapter preparation, network, server inference, and response
handling; it is not pure model latency. Outside-invoke time includes extra
network work as well as local processing.

Cold cache inventories: direct 1,050 new .pyc files per round; Akasha 1,084,
only 34 additional files (3.2%). Warm runs created zero new .pyc files.
After the answer, module counts were direct 1,210 vs Akasha 1,248. Extra
non-Akasha modules included dotenv, jieba3, opencc, and Core document types.
This replaces the misleading FakeListLLM-versus-Gemini comparison when
discussing the incremental cost of the current Akasha wrapper.

## Additional Work Located

Source inspection:

- `akasha/tools/ask.py` calls get_doc_length and myTokenizer.compute_tokens
  before generation. Chinese get_doc_length initializes jieba3 resources.
- `akasha/helper/token_counter.py:compute_tokens_gemini` delegates to
  `akasha/utils/models/gemi.py:calculate_token`.
- calculate_token creates a NEW `google.genai.Client`, then sends a remote
  `client.models.count_tokens` request. Direct adapter invoke does not do this.
- `akasha/helper/run_llm.py:call_model` subsequently applies sim_to_trad,
  initializing OpenCC in each fresh process on first use.

Supplementary boundary probe: three fresh processes with existing bytecode,
normal public ask path, only generation invoke replaced by a fixed AIMessage.
Token counting remained REAL and was timed around calculate_token. This is
not a substitute for the twelve real generation measurements above; it isolates
non-generation work and adds three real count_tokens requests.

| Supplementary work | Round 1 | Round 2 | Round 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Chinese tokenizer getter | 0.887 s | 0.857 s | 0.854 s | 0.857 s |
| OpenCC getter | 0.046 s | 0.043 s | 0.043 s | 0.043 s |
| calculate_token: new client + count API | 1.356 s | 1.317 s | 1.314 s | 1.317 s |
| Entire call with generation stubbed | 2.289 s | 2.218 s | 2.213 s | 2.218 s |

One real count_tokens call per supplementary probe, no count errors/fallbacks.
An earlier standalone resource-only probe gave about 1.05 s for tokenizer plus
converter initialization. Do not add that standalone timing to the boundary
probe; it measures the same resources under a different process/import state.

## Conclusions and Limits

1. Direct LangChain with the SAME Gemini adapter is also slow with empty
   bytecode in this environment: about 17-18 seconds to the first answer here.
   The baseline is not a near-instant SDK/fake-model application.
2. Current Akasha adds relatively few bytecode files, not another comparable
   dependency tree. Startup remains sensitive to underlying I/O variability.
   This run did not trace syscalls and cannot allocate each cold outlier.
3. A repeatable incremental cost exists outside the generation adapter,
   about 2.3 seconds in the warm real-call runs. The supplementary probe
   locates most of it in remote token counting/new-client setup and Chinese
   resource initialization, rather than bytecode cache renaming.
4. Remote token counting is per-question work, not inherently a one-time
   startup cache. Chinese resources are cached within a process, not between
   fresh processes. Reusing a long-lived ask client therefore helps the latter,
   but does not by itself remove count_tokens requests.
5. Optimizing input budgeting or Chinese initialization requires a separate
   behavior-preserving design/test decision. No token checks or conversion
   were disabled in the real-call benchmark or changed in production.

Bytecode-cold is not machine/OS-cache cold; the same installed environment
contains unused packages. Network variability remains. These results apply
to the fixed short prompt, Chinese mode, Gemini API-key path, and these package
versions, not arbitrary RAG/agent/document workloads or all installations.

## Evidence and Reproduction

```powershell
.venv/Scripts/python.exe scripts/benchmark_adapter_vs_akasha.py `
  --output artifacts/startup-profile/NEW-ADAPTER-COMPARISON --repeats 3
```

Sends twelve real generation requests with credentials from tests/.env or
environment. Akasha also performs remote token counting. Use a fresh output
directory. Existing valid local configuration is required; no keys in command.

Artifacts: `artifacts/startup-profile/adapter-vs-akasha-live-20261009/`:
`runs.json`, `summary.json`, per-child results, `chinese-resource-probe.json`,
and `outside-adapter-probe.json`. All generation request hashes matched;
all smoke responses were exact; all warm .pyc inventory deltas were zero.
