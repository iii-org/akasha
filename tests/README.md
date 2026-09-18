# Tests

The suite is organized by product feature. See
[TESTING_GUIDE.md](TESTING_GUIDE.md) for marker rules, live-service behavior,
CI policy, and the coverage map.

Install the [shared dependency baseline](../constraints/README.md) first.
From the repository root in PowerShell:

```powershell
$python = ".venv\Scripts\python.exe"

& $python -m pytest --collect-only -q
& $python -m pytest -m unit -q
& $python -m pytest -m integration -q
& $python -m pytest -q
```

External-service tests use one explicit switch:

```powershell
$env:RUN_LIVE_TESTS = "1"
& $python -m pytest -m live -q
```

Each provider case skips independently when its credential is absent. Ollama
also requires an explicit, reachable `OLLAMA_API_BASE`.
