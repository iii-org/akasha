"""Compare empty-cache startup with install-time compileall preparation."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]


def environment(cache):
    prefix = "\\\\?\\" + str(cache.resolve()) if os.name == "nt" else str(cache.resolve())
    env = dict(os.environ, PYTHONPYCACHEPREFIX=prefix,
               LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
    for name in ("PYTHONDONTWRITEBYTECODE", "AKASHA_STARTUP_CACHE_OPS"):
        env.pop(name, None)
    return env


def inventory(cache):
    if os.name == "nt":
        cache = Path("\\\\?\\" + str(cache.resolve()))
    files = list(cache.rglob("*.pyc")) if cache.exists() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume-precompiled", action="store_true",
                        help="Reuse a completed preparation if measurement has not started")
    args = parser.parse_args()
    if (args.output.exists() and not args.resume_precompiled) or args.repeats < 1:
        parser.error("Use a new output directory and positive repeats")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume_precompiled)
    prepared = output / "cache-precompiled"
    env = environment(prepared)
    stages = []
    if args.resume_precompiled:
        stages = json.loads((output / "precompile.json").read_text(encoding="utf-8"))
        if len(stages) != 3 or not prepared.exists() or (output / "runs.json").exists():
            parser.error("Resume requires complete preparation and no existing measurements")
    # Include stdlib: compiling only site-packages leaves first-use stdlib writes.
    roots = [("stdlib", Path(sysconfig.get_path("stdlib"))),
             ("site-packages", Path(sysconfig.get_path("purelib"))),
             ("akasha", ROOT / "akasha")]
    for label, path in ([] if args.resume_precompiled else roots):
        excludes = r"[\\/](tests?|site-packages)[\\/]" if label == "stdlib" else r"[\\/]tests?[\\/]"
        command = [sys.executable, "-m", "compileall", "-q", "-x", excludes, str(path)]
        start = time.perf_counter()
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
        elapsed = time.perf_counter() - start
        if completed.returncode:
            print(completed.stdout, flush=True)
            raise RuntimeError(f"compileall failed for {label}; exit={completed.returncode}")
        stages.append({"stage": label, "path": str(path), "seconds": elapsed})
        print("PRECOMPILE", label, round(elapsed, 3), flush=True)
        (output / "precompile.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
    initial_inventory = inventory(prepared)
    print("PRECOMPILE_FILES", initial_inventory, flush=True)
    records = []
    for run in range(1, args.repeats + 1):
        order = ("cold", "precompiled") if run % 2 else ("precompiled", "cold")
        for condition in order:
            cache = output / f"cache-cold-{run}" if condition == "cold" else prepared
            before = inventory(cache)
            result_path = output / f"{condition}-{run}.json"
            command = [sys.executable, str(ROOT / "scripts/benchmark_feature_startup.py"),
                       "--child", "--source", str(ROOT), "--output", str(result_path)]
            if args.live:
                command.append("--live")
            start = time.perf_counter()
            completed = subprocess.run(command, env=environment(cache), capture_output=True,
                                       text=True, timeout=180)
            wall = time.perf_counter() - start
            if completed.returncode:
                raise RuntimeError(f"{condition}/{run} failed; exit={completed.returncode}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if args.live and not result["phases"].get("response_exact"):
                raise RuntimeError(f"Unexpected smoke response for {condition}/{run}")
            after = inventory(cache)
            result.update(condition=condition, run=run, process_wall_seconds=wall,
                          cache_before=before, cache_after=after,
                          new_pyc_files=after["files"] - before["files"])
            records.append(result)
            print(condition, run, result["phases"], "new_pyc_files", result["new_pyc_files"], flush=True)
            (output / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"precompile_stages": stages,
               "precompile_seconds": sum(s["seconds"] for s in stages),
               "initial_precompiled_cache": initial_inventory,
               "live": args.live, "repeats": args.repeats,
               "note": "New processes; empty bytecode cache is not an empty OS disk cache. No import hooks."}
    for condition in ("cold", "precompiled"):
        group = [r for r in records if r["condition"] == condition]
        summary[condition] = {
            metric: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for metric in group[0]["phases"] if metric.endswith("_seconds")
            for values in [[r["phases"][metric] for r in group]]}
        summary[condition]["new_pyc_files"] = [r["new_pyc_files"] for r in group]
        summary[condition]["process_wall_seconds"] = [r["process_wall_seconds"] for r in group]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
