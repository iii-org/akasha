# Install-Time Precompilation: Live Startup Comparison

Date: 2026-10-09. Current modified Akasha source and existing light test
environment, Python 3.11.11. Torch, Transformers, FAISS, and MLflow were absent.
This simulates the preparation step after installation; it does not reinstall
packages or measure dependency download/resolution/install time.

## Method

Prepared a new isolated bytecode prefix using the environment's Python and
`python -m compileall -q`. Compiled stdlib, existing site-packages, and local
Akasha source. Excluded test directories and excluded site-packages from the
stdlib pass. Compilation does not execute the target modules or send API calls.

Used `scripts/benchmark_precompiled_startup.py --live --repeats 3` and the
existing live benchmark child. Six sequential real Gemini requests, each in a
new Python process. Alternated cold/precompiled order between rounds.

- Cold: a different empty bytecode prefix for each round.
- Precompiled: reuse the preparation prefix, without first running Akasha.
- Same current source, interpreter, credentials, Gemini 2.5 Flash, Chinese
  language default, max_output_tokens=128, and prompt requesting ASK_SMOKE_OK.
- No import hooks, cProfile, or native I/O tracing. LangSmith service tracing off.
- Bytecode writes allowed in both conditions; no PYTHONDONTWRITEBYTECODE shortcut.
- Timed import, first construction, and first actual answer. The main timer
  excludes interpreter/harness startup and process teardown; whole-process
  wall time was recorded separately.

This is a bytecode-cold comparison, not a machine/OS-disk-cache cold comparison.
Precompilation also reads source files; OS cache state is not controlled.
Three rounds bound this experiment, not all machines or all models.

## Live Results

All six requests returned exactly the expected smoke response.

| First-answer time | Round 1 | Round 2 | Round 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Empty cache | 23.209 s | 25.645 s | 23.521 s | 23.521 s |
| Precompiled | 4.765 s | 4.840 s | 4.821 s | 4.821 s |

Median first-answer reduction: 18.700 seconds, approximately 79.5%.

| Phase | Empty cache median | Precompiled median |
| --- | ---: | ---: |
| Import | 13.783 s | 0.841 s |
| Construct first ask instance | 6.628 s | 1.411 s |
| Ready (import + construction) | 20.200 s | 2.253 s |
| First call, including deferred resources/network/model | 3.286 s | 2.571 s |
| First answer, including ready | 23.521 s | 4.821 s |

Phase medians are independent; their sums need not equal the median total.
Whole-process wall times were 24.756-27.061 s for cold and 5.103-5.182 s
for precompiled. The gain therefore also exists outside the internal timer.

Cold children created 1,094 new .pyc files each. All three precompiled children
created ZERO new .pyc files. This supports actual reuse of the prepared cache,
not just reusing modules in the same process. First-call timing also includes
Chinese resources and network; do not attribute its entire difference to .pyc.

## Preparation Cost

| Preparation stage | Elapsed |
| --- | ---: |
| Standard library | 148.920 s |
| Existing site-packages | 648.124 s |
| Local Akasha source | 1.778 s |
| Total | 798.822 s (13 min 19 s) |

Prepared 15,928 bytecode files, totaling 218,125,369 bytes (about 218 MB).
The site-packages tree contains roughly 17,000 Python source files; excluding
tests and already compiled bootstrap modules affects the output count.

This broad preparation is much larger than a first-answer dependency path.
It moves cost out of the first interaction but does not eliminate it. The
13-minute measurement is NOT an estimate for every pip/uv install or for
compiling only Akasha's required dependency closure. No compiler profiling
was enabled, so this run does not allocate preparation time between CPU and I/O.

## Interpretation

Install-time precompilation can materially improve first-answer latency in
this environment. Keeping Python version, bytecode invalidation settings,
source timestamps/content, and cache location compatible is necessary.
Source/package updates can require fresh compilation.

Do not automatically add a full-environment recursive compileall pass to the
normal installation flow based solely on the startup gain. Its preparation
cost and disk footprint are substantial here. Next compare a smaller declared
precompile scope or installer-supported bytecode preparation, including
first-answer .pyc misses and installation cost. Stdlib is a separate concern:
compiling only installed third-party packages does not necessarily cover it.

No production code, installation requirements, .env, or package versions
were changed in this experiment. The benchmark is a diagnostic script, not
an installed-package post-install hook.

## Reproduction and Evidence

```powershell
.venv/Scripts/python.exe scripts/benchmark_precompiled_startup.py `
  --output artifacts/startup-profile/NEW-PRECOMPILE-COMPARISON `
  --repeats 3 --live
```

This sends six small real API requests and requires local configured credentials.
Use a NEW output directory to preserve cold-cache isolation.

Artifacts: `artifacts/startup-profile/install-precompiled-live-20261009/`.
`precompile.json`, `runs.json`, and `summary.json` contain costs, per-round
phase timings, cache inventories, and validity flags. Individual child result
files retain interpreter/model settings; no keys or response bodies are saved.

The first harness invocation completed all preparation stages, then failed
while inventorying Windows long paths. Fixed inventory to use extended paths
and resumed from the completed preparation using `--resume-precompiled`.
No API measurements had begun before this fix; no precompile work was repeated
or omitted from the reported preparation cost.
