"""Compare core-only, direct SDK and Akasha startup with isolated caches."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = ("core", "direct", "akasha")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in range(1, args.repeats + 1):
        offset = (run - 1) % len(CASES)
        for case in CASES[offset:] + CASES[:offset]:
            env = os.environ.copy()
            env["PYTHONPYCACHEPREFIX"] = "\\\\?\\" + str(output / f"{case}-{run}-pyc")
            env.pop("PYTHONDONTWRITEBYTECODE", None)
            env["AKASHA_STARTUP_CACHE_OPS"] = "1"
            env["LANGSMITH_TRACING"] = "false"
            env["LANGCHAIN_TRACING_V2"] = "false"
            for temperature in (("cold", "warm") if case == "core" else ("cold",)):
                directory = output / temperature
                directory.mkdir(exist_ok=True)
                prefix = directory / f"{case}-{run}"
                command = [sys.executable, "-u", str(ROOT / "scripts/profile_startup.py"),
                           "--case", case, "--run", str(run), "--trace-imports",
                           "--threshold", "120", "--output", str(directory)]
                print(f"RUN {run} {case} {temperature}", flush=True)
                with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=180)
                data = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
                trace = data["import_trace"]
                files = trace["files"]
                operations = trace["cache_operations"]
                writes = sum(f["writes"] for f in files)
                for name in ("open", "write", "close", "replace"):
                    assert sum(e["operation"] == name for e in operations) == writes
                modules = [m["module"] for m in trace["modules"]]
                assert case == "akasha" or not any(m == "akasha" or m.startswith("akasha.") for m in modules)
                row = {"case": case, "run": run, "temperature": temperature,
                       "first_ready_s": sum(p["seconds"] for p in data["phases"]
                                            if p["name"] != "second_construct"),
                       "writes": writes, "cache_s": sum(f["cache_s"] for f in files),
                       "compile_s": sum(f["compile_s"] for f in files),
                       "slow_ops_100ms": sum(e["seconds"] >= 0.1 for e in operations),
                       "maximum_op_s": max((e["seconds"] for e in operations), default=0),
                       "langsmith_loaded": any(m == "langsmith" for m in modules),
                       "akasha_loaded": any(m == "akasha" for m in modules),
                       "write_errors": sum(len(f.get("write_errors", [])) for f in files),
                       "operations": {name: sum(e["seconds"] for e in operations if e["operation"] == name)
                                      for name in ("open", "write", "close", "replace", "mkdir", "fd_wrap")}}
                rows.append(row)
                (output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)
    for case, temperature in [(c, "cold") for c in CASES] + [("core", "warm")]:
        group = [r for r in rows if r["case"] == case and r["temperature"] == temperature]
        print("MEDIAN", case, temperature, json.dumps({key: statistics.median(r[key] for r in group)
              for key in ("first_ready_s", "cache_s", "writes", "maximum_op_s")}))


if __name__ == "__main__":
    main()
