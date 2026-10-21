"""Offline LangSmith import/cache attribution; not a production workaround."""

import argparse
from importlib import _bootstrap, _bootstrap_external
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("baseline", "skip-own-cache", "skip-branch-cache")


def child(args):
    from scripts import profile_startup

    original_load = _bootstrap._load_unlocked
    original_cache = _bootstrap_external.SourceFileLoader._cache_bytecode
    depth = 0
    branch_modules = set()
    intervals = []
    skipped = []

    def load(spec):
        nonlocal depth
        smith = spec.name == "langsmith" or spec.name.startswith("langsmith.")
        root = smith and depth == 0
        entered = smith or depth > 0
        if entered:
            depth += 1
            branch_modules.add(spec.name)
        started = time.perf_counter()
        try:
            return original_load(spec)
        finally:
            if root:
                intervals.append({"module": spec.name,
                                  "seconds": time.perf_counter() - started})
            if entered:
                depth -= 1

    def cache(loader, *a, **kw):
        own = loader.name == "langsmith" or loader.name.startswith("langsmith.")
        skip = (args.mode == "skip-own-cache" and own) or (
            args.mode == "skip-branch-cache" and depth > 0)
        if skip:
            skipped.append(loader.name)
            return None
        return original_cache(loader, *a, **kw)

    _bootstrap._load_unlocked = load
    _bootstrap_external.SourceFileLoader._cache_bytecode = cache
    try:
        profile_startup.child(argparse.Namespace(
            case="akasha", run=args.run, model="gemini-2.5-flash",
            output=args.output, profile=False, trace_imports=True,
            stack_after=5, threshold=120))
    finally:
        _bootstrap._load_unlocked = original_load
        _bootstrap_external.SourceFileLoader._cache_bytecode = original_cache
    path = args.output / f"akasha-{args.run}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    files = result["import_trace"]["files"]
    branch_files = [f for f in files if f["module"] in branch_modules]
    result["langsmith_probe"] = {
        "mode": args.mode, "intervals": intervals,
        "inclusive_s": sum(i["seconds"] for i in intervals),
        "branch_modules": sorted(branch_modules), "skipped": skipped,
        "branch_compile_s": sum(f["compile_s"] for f in branch_files),
        "branch_cache_s": sum(f["cache_s"] for f in branch_files),
        "branch_writes": sum(f["writes"] for f in branch_files),
        "own_cache_s": sum(f["cache_s"] for f in files if f["package"] == "langsmith"),
        "own_writes": sum(f["writes"] for f in files if f["package"] == "langsmith"),
        "total_writes": sum(f["writes"] for f in files),
        "write_errors": sum(len(f.get("write_errors", [])) for f in files),
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode:
        child(args)
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = []
    for run in range(1, args.repeats + 1):
        order = MODES[(run - 1) % 3:] + MODES[:(run - 1) % 3]
        for mode in order:
            directory = args.output / mode
            directory.mkdir(exist_ok=True)
            prefix = args.output / f"{mode}-{run}-pyc"
            if prefix.exists():
                raise RuntimeError(f"Cold cache already exists: {prefix}")
            env = os.environ.copy()
            env.pop("PYTHONDONTWRITEBYTECODE", None)
            env["PYTHONPYCACHEPREFIX"] = "\\\\?\\" + str(prefix)
            # No model invocation; force tracing off even if the shell enables it.
            env["LANGSMITH_TRACING"] = "false"
            env["LANGCHAIN_TRACING_V2"] = "false"
            temperatures = ("cold", "warm") if mode == "baseline" else ("cold",)
            for temperature in temperatures:
                output = directory / temperature
                output.mkdir(exist_ok=True)
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--mode", mode, "--run", str(run), "--output", str(output)]
                print(f"RUN {run} {mode} {temperature}", flush=True)
                with (output / f"akasha-{run}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=120, check=True)
                result = json.loads((output / f"akasha-{run}.json").read_text(encoding="utf-8"))
                row = dict(result["langsmith_probe"], run=run, temperature=temperature,
                           first_ready_s=sum(p["seconds"] for p in result["phases"]
                                             if p["name"] != "second_construct"))
                row.pop("branch_modules")
                row.pop("intervals")
                row["skipped"] = len(row["skipped"])
                rows.append(row)
                print(json.dumps(row), flush=True)
    medians = []
    for mode, temperature in [(m, "cold") for m in MODES] + [("baseline", "warm")]:
        group = [r for r in rows if r["mode"] == mode and r["temperature"] == temperature]
        medians.append(dict(mode=mode, temperature=temperature, **{
            key: statistics.median(r[key] for r in group)
            for key in ("first_ready_s", "inclusive_s", "branch_cache_s", "own_cache_s",
                        "branch_writes", "own_writes", "total_writes", "write_errors")}))
    (args.output / "summary.json").write_text(
        json.dumps({"runs": rows, "medians": medians}, indent=2), encoding="utf-8")
    print("MEDIANS " + json.dumps(medians), flush=True)


if __name__ == "__main__":
    main()
