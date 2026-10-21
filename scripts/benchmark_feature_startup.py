"""Compare source snapshots sequentially, optionally including a real Ask call."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def child(args):
    sys.path.insert(0, str(args.source.resolve()))
    if not args.live:
        os.environ["GEMINI_API_KEY"] = "startup-diagnostic-placeholder"
        os.environ.pop("GOOGLE_API_KEY", None)
        for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT",
                     "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ.pop(name, None)
    phases = {}
    start = time.perf_counter()
    import akasha
    ask = akasha.ask
    phases["import_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    qa = ask(model=f"gemini:{args.model}", env_file=str(args.env_file) if args.live else "",
             max_output_tokens=128, thinking_budget=0)
    phases["construct_seconds"] = time.perf_counter() - start
    phases["ready_seconds"] = phases["import_seconds"] + phases["construct_seconds"]
    if args.live:
        start = time.perf_counter()
        response = qa("Reply with exactly: ASK_SMOKE_OK")
        phases["first_call_seconds"] = time.perf_counter() - start
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Ask returned no visible text")
        phases["first_total_seconds"] = phases["ready_seconds"] + phases["first_call_seconds"]
        phases["response_valid"] = True
        phases["response_exact"] = response.strip() == "ASK_SMOKE_OK"
    result = {
        "source": str(args.source.resolve()), "python": sys.version,
        "model": args.model, "live": args.live, "language": "ch",
        "max_output_tokens": 128, "thinking_budget": 0,
        "phases": phases, "module_count": len(sys.modules),
        "feature_modules": sorted(name for name in sys.modules if name.startswith("akasha.")),
        "note": "Excludes interpreter/harness startup; first call includes deferred resources and network.",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(phases), flush=True)


def compare(args):
    if args.output.exists():
        raise ValueError("Use a new output directory so cold bytecode caches stay empty")
    args.output.mkdir(parents=True)
    records = []
    sources = {"before": args.before.resolve(), "after": ROOT}
    for run in range(1, args.repeats + 1):
        order = ("before", "after") if run % 2 else ("after", "before")
        for label in order:
            cache = (args.output / f"cache-{label}-{run}").resolve()
            prefix = "\\\\?\\" + str(cache) if os.name == "nt" else str(cache)
            env = {**os.environ, "PYTHONPYCACHEPREFIX": prefix}
            env.pop("PYTHONDONTWRITEBYTECODE", None)
            env["LANGCHAIN_TRACING_V2"] = "false"
            env["LANGSMITH_TRACING"] = "false"
            for condition in ("cold", "warm"):
                output = args.output / f"{label}-{condition}-{run}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--child",
                           "--source", str(sources[label]), "--output", str(output),
                           "--model", args.model, "--env-file", str(args.env_file.resolve())]
                if args.live:
                    command.append("--live")
                completed = subprocess.run(command, env=env, capture_output=True,
                                           text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(f"{label}/{condition}/{run} failed; exit={completed.returncode}")
                result = json.loads(output.read_text(encoding="utf-8"))
                result.update(label=label, condition=condition, run=run)
                records.append(result)
                print(label, condition, run, result["phases"], flush=True)
                (args.output / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {}
    for label in sources:
        for condition in ("cold", "warm"):
            group = [r["phases"] for r in records if r["label"] == label and r["condition"] == condition]
            summary[f"{label}_{condition}"] = {
                metric: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                for metric in group[0] if metric.endswith("_seconds")
                for values in [[row[metric] for row in group]]
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--env-file", type=Path, default=ROOT / "tests/.env")
    parser.add_argument("--live", action="store_true", help="Send one small API request per child")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.child:
        child(args)
    else:
        if args.before is None or not (args.before / "akasha/__init__.py").is_file():
            parser.error("before must contain a pre-change akasha source snapshot")
        compare(args)


if __name__ == "__main__":
    main()
