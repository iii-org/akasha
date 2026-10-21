# Post-Lazy-Loading Bytecode Cache Diagnosis

Date: 2026-10-09. Scope: current modified source, existing light environment.
No production changes or API calls in this investigation.

## Question and Method

Does remaining startup latency still come from creating, writing, closing,
and renaming Python bytecode cache files?

Used `scripts/profile_startup.py --case akasha --trace-imports` with
`AKASHA_STARTUP_CACHE_OPS=1`. Instrumented CPython's actual cache operations,
without replacing its atomic-write algorithm. Tracing was disabled for
LangChain/LangSmith services. Each measurement used a new process.

- Cold: unique, empty `PYTHONPYCACHEPREFIX` on the workspace volume.
- Warm: reuse that cold run's bytecode prefix in a new process.
- No-write: empty prefix plus `PYTHONDONTWRITEBYTECODE=1`; source must still compile.
- Three rounds, sequential; alternated warm/no-write order after each cold run.
- Extended Windows paths avoided long-path cache failures. All cold writes succeeded.

The timer covers Akasha import and first offline `ask` construction, not a
real answer, interpreter bootstrap, or second construction. Empty bytecode
cache is not a cold OS filesystem cache or a fresh machine. These measurements
are not directly comparable to the earlier live-answer benchmark.

## Results

| Condition | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Empty cache, writes enabled | 16.626 s | 10.557 s | 15.608 s | 15.608 s |
| Reused cache, new process | 2.426 s | 4.830 s | 2.587 s | 2.587 s |
| Empty cache, writes disabled | 4.401 s | 4.366 s | 4.381 s | 4.381 s |

Each cold run wrote 1,007 files, totaling 17,970,095 bytes, with zero errors.
No-write and warm conditions recorded zero cache writes.

| Cold timing component | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Source compilation | 1.887 s | 2.005 s | 1.942 s |
| Complete bytecode cache workflow | 11.814 s | 5.873 s | 11.026 s |
| Remaining measured startup | 2.925 s | 2.679 s | 2.640 s |
| Cache workflow / startup | 71.1% | 55.6% | 70.6% |

These three components are additive. Cache operations below are nested within
the complete cache workflow; do not add them again to startup totals.

| Native cache operation | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Open/create temporary file | 4.807 s | 3.723 s | 5.430 s |
| Write bytes | 4.978 s | 0.166 s | 3.160 s |
| Close file | 0.259 s | 0.148 s | 0.228 s |
| Replace/rename to final .pyc | 1.120 s | 1.225 s | 1.322 s |
| Create directories | 0.396 s | 0.352 s | 0.632 s |

Each run had 1,007 open/write/close/replace operations and 141 mkdir attempts.
The workflow also contains marshaling, directory checks, wrapper construction,
and tracing overhead. Ordinary per-operation medians were about 0.225 ms for
open, 0.124-0.129 ms for write, and 0.98-1.01 ms for replace; totals are strongly
affected by slow-tail operations.

## Slow Tails and Package Attribution

- Run 1: `httpcore._trace` write 3.519 s; Pydantic namespace utilities write 1.322 s.
- Run 2: `httpcore._models` open 0.548 s; `requests.hooks` open 0.261 s.
- Run 3: OpenTelemetry `otel_attributes` write 2.915 s; urllib3 open 0.381 s.

Cache-only package times, not inclusive import times:

| Package | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| langchain_core | 0.792 s | 0.795 s | 0.848 s |
| langsmith | 0.516 s | 0.540 s | 0.856 s |
| aiohttp | 0.300 s | 0.146 s | 0.427 s |

Large outliers move between packages. A slow .pyc operation does not prove
that the corresponding package deliberately implements an expensive cache.
This is CPython's import cache, not LangSmith service caching or an API request.

## Uninstrumented Control

One additional sequential round without import hooks or native-operation
instrumentation: cold 12.544 s, warm 3.416 s, no-write 4.428 s.
The large write-enabled vs no-write gap remains without hooks. One control
round does not quantify tracing overhead separately from I/O variability.

## Interpretation and Limits

The remaining empty-bytecode-cache cost is primarily cache creation I/O,
not source compilation. Disabling writes retains compilation but removes
roughly 6-12 seconds in these trials. Reusing valid cache avoids both costs.
Import breadth still matters: over a thousand cache writes expose startup
to cumulative operation costs and sporadic long waits. Deferred optional
features do not remove the required provider/core dependency imports.

We measured elapsed calls at the Python/native I/O boundary, not physical
disk service time. This turn did not recapture Windows ETW, identify a filter
driver, establish antivirus causality, or rule out scheduling/storage effects.
Previous kernel tracing is historical supporting evidence, not a new capture.
Warm run 2 also shows that unrelated read/initialization stalls can remain.

Do not ship `PYTHONDONTWRITEBYTECODE=1` as a blanket fix: it forces recompilation
on future launches when no valid cache exists. The next useful controlled
experiment is the same dependency tree on a verified SSD/cache location or
install-time bytecode precompilation, followed by first-answer measurements.
Security-software exclusions require a separate approved experiment.

## Evidence and Reproduction

Raw JSON and additive summaries:
`artifacts/startup-profile/post-lazy-cache-20261009/`

- `cold/akasha-{1,2,3}.json`: source, package, and individual syscall timings.
- `cold/operation-summary.json`: operation counts and slow-tail events.
- `warm/`, `no-write/`: the paired controls.
- `diagnosis-summary.json`: per-run additive totals and medians.
- `untraced-cold/`, `untraced-warm/`, `untraced-no-write/`: no-hook controls.

Set a new, empty extended-path `PYTHONPYCACHEPREFIX`, then run:

```powershell
$env:AKASHA_STARTUP_CACHE_OPS = '1'
.venv/Scripts/python.exe scripts/profile_startup.py --case akasha `
  --run 1 --output artifacts/startup-profile/NEW-DIAGNOSTIC/cold `
  --threshold 10 --trace-imports
```

Exit code 1 indicates the 10-second startup threshold was exceeded, not an
import failure. For no-write set `PYTHONDONTWRITEBYTECODE=1` and a different
empty prefix. For warm remove that flag and reuse the preceding cold prefix.
