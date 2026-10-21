# UV installation bytecode comparison

Date: 2026-10-09. Current Akasha source, light dependencies, Windows.

## Results

Seconds, medians of three fresh environments per condition:

| Measurement | Normal install | --compile-bytecode |
| --- | ---: | ---: |
| Installation | 114.34 | 241.95 |
| First process: imports and construction, before ask | 26.24 | 2.56 |
| First process: through complete real API answer | 29.42 | 5.23 |
| Second new process: through complete real API answer | 5.29 | 4.95 |
| Install plus first answer, per-run combined median | 135.53 | 247.29 |

Installation median increased 127.61 seconds. First-answer median decreased
24.18 seconds (about 82%). Combined medians are computed from per-run sums,
not sums of independent medians. Precompilation shifts work to installation;
it did not reduce total installation plus first-answer wait on this machine.

| Pair | Normal install | Compiled install | Normal first answer | Compiled first answer |
| --- | ---: | ---: | ---: | ---: |
| 1 | 75.59 | 276.95 | 29.42 | 5.02 |
| 2 | 147.08 | 229.81 | 38.21 | 5.23 |
| 3 | 114.34 | 241.95 | 21.18 | 5.34 |

## Method and validation

- Python 3.11.11; uv 0.12.17 (635500036, 2026-09-18).
- Six fresh virtual environments; three alternating normal/compiled pairs.
- Identical 160 installed distributions: pinned light dependencies and the
  same version 1.8.3 wheel built from the current modified Akasha source.
- Installs used --no-deps --no-config --no-progress --offline; only the compiled
  condition added --compile-bytecode. UV distribution cache was primed first.
- Separate initial cache preparation install: 147.83 seconds, excluded from
  measured install times. Resolution, wheel build and venv creation excluded.
- This measures cached/offline installation, not first-ever download/install.
  Existing Python standard-library caches and Windows disk caches were shared;
  this is not reboot-level cold startup.
- Default Python __pycache__ locations. Inherited prefix/no-write/UV compilation
  settings removed. Each environment ran two separate processes.
- Startup timer includes Akasha import, construction and a real Gemini 2.5 Flash
  answer, but excludes Python/harness bootstrap. Whole process wall times saved.
- All twelve generation responses matched ASK_SMOKE_OK. Credentials came from
  tests/.env; OS GEMINI_API_KEY was absent. The token-count helper can fall back
  to local counting in this configuration. This round does not establish remote
  count_tokens timing. Both conditions used the same configuration.
- All six version manifests matched; no Torch, Transformers, FAISS or MLflow.
  uv pip check passed for the final compiled environment. Isolated import
  confirmed the installed wheel path, not the source checkout.
- An initial harness attempt failed parsing manifest JSON before API calls;
  excluded. Corrected runner used fresh environments and reused prepared assets.
- No production behavior or original .venv was changed by this benchmark.

Every normal install had one bootstrap .pyc (14,211 bytes). Every compiled install
had 17,210 site-packages .pyc files (263,474,203 bytes, about 263 MB).
Normal first starts each added 901 .pyc files; compiled first starts added zero.
All second starts added zero. These inventories demonstrate that the flag
actually produced caches reused during startup.

One UV log reported installation in 1m 19s and bytecode compilation of 17,214
files in 3m 16s. UV's count and filesystem .pyc inventory are distinct metrics;
the four-file difference was not explained. Unlike the earlier 13-minute manual
compileall test, this UV experiment did not compile the standard library as
part of installation and used a differently scoped environment. Median total
UV installation here was about four minutes.

## Cache location and underlying I/O

Python supports PYTHONPYCACHEPREFIX or -X pycache_prefix=PATH to store bytecode
in a separate mirrored tree. Configure before Python starts and use the same
effective prefix when precompiling and running. Prefer a launcher/deployment
setting over hidden process-wide behavior inside Akasha. UV_CACHE_DIR instead
controls the distribution cache, not Python runtime bytecode.

A no-dependencies Akasha-wheel probe used UV precompilation with the short
prefix C:\tmp\ak-pyc-20261009. There were 86 Akasha .pyc files before importing
the installed package and 86 afterward. Runtime confirmed the prefix and
installed package path. This validates compatibility for this UV/Python version,
not SSD speed or a full ask run with the alternate prefix.

An earlier deeply nested extended Windows prefix yielded only five observed
Akasha .pyc files after import despite UV reporting 87 compiled files. Its cause
was not established; do not assume arbitrary long prefixes work unchanged.

Hardware inspection: C and D are partitions on the same WDC SATA HDD (Disk 0).
Other listed devices were USB flash and USB HDD; no confirmed SSD destination
was available. C-to-D migration does not move to another physical disk.
No disk mounts, source moves, antivirus exclusions or driver changes were made.

Akasha can reduce imports and move bytecode writes to installation, but cannot
directly repair disk/driver/filter latency. A genuinely faster local cache disk
is testable; source/DLL reads and initialization still remain. Measurements do
not prove antivirus or a specific driver caused prior stalls.

Reference: https://docs.python.org/3.11/using/cmdline.html#envvar-PYTHONPYCACHEPREFIX
UV option verified with the local version's uv pip install --help.

## Reproduction and artifacts

Runner: scripts/benchmark_uv_install.py.

```powershell
.venv/Scripts/python.exe scripts/benchmark_uv_install.py `
  --output artifacts/startup-profile/uv-install-measured-20261009 `
  --environments .uvbench/20261009b `
  --prepared-assets artifacts/startup-profile/uv-install-20261009 `
  --repeats 3
```

Use new output/environment paths for a new experiment. Generated environments
are ignored under .uvbench. Records:

- artifacts/startup-profile/uv-install-measured-20261009/runs.json: six installs,
  twelve startup measurements, package manifests and bytecode inventories.
- Same directory summary.json and combined-summary.json: calculated medians.
- artifacts/startup-profile/uv-install-20261009/: wheel, lock and setup evidence.
- artifacts/startup-profile/uv-prefix-probe-20261009/short-prefix-result.json:
  successful short-prefix compatibility probe.
- Same directory result.json: initial long-prefix observation.
