# Shared development and CI dependency baseline

`uv.lock` is the source of truth for resolved development dependencies across
Python 3.11/3.12. Both profiles are exported from the same lock, so shared
packages have the same versions. `light-dev.txt` includes light + dev;
`full-dev.txt` includes full + dev. Platform/Python markers are retained.

Use **uv 0.12.13**, the same resolver version bootstrapped in CI. From the repo
root, install into a new project-local environment:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[light,dev]" -c constraints/light-dev.txt
uv pip check --python .venv/Scripts/python.exe
```

For full mode, replace `light` with `full` and use `constraints/full-dev.txt`.
Full still needs the documented native build toolchain; the lock does not remove
those requirements. Installing a full profile into a former light environment
adds optional packages; use separate environments to test the light boundary.
The older `..\.venv` environment is preserved but is no longer the CI baseline.

The SDK baseline is OpenAI 3.16.2, langchain-openai 1.6.2, HTTPX2 2.13.0 and
HTTPCore2 2.13.0. Legacy HTTPX remains installed for other dependencies; its
presence is not an error. Model construction uses the SDK's compatible
DefaultAsyncHttpxClient helper, which now creates an HTTPX2 client.

Public package metadata declares compatible version ranges rather than pinning
all downstream users. `requirements.txt` and `requirements-light.txt` remain
generated public dependency ranges; they are not the development baseline.
Use these constraints when reproducibility is required.

## Updating the baseline

Update intentionally, review the lock and exported diffs, and rerun local and
live provider tests. Do not independently regenerate profiles with pip compile:
that can select incompatible shared versions.

```powershell
uv lock --upgrade-package openai --upgrade-package langchain-openai --upgrade-package httpx2 --upgrade-package httpcore2
uv export --locked --extra light --extra dev --no-emit-project --no-hashes -o constraints/light-dev.txt
uv export --locked --extra full --extra dev --no-emit-project --no-hashes -o constraints/full-dev.txt
```

When changing public dependency ranges, also run
`python scripts/sync_requirements.py`. CI uses `--locked`, verifies exports, installs
with constraints, and runs `uv pip check`; it fails if the manifest and lock drift.
The lock records artifact hashes; pip constraint exports pin versions without
requiring hashes because the project is installed editable.

See the [validation report](../dev_docs/2026-09-httpx2-dependency-baseline.md)
for fresh-environment, provider and Linux resolver results and remaining limits.
