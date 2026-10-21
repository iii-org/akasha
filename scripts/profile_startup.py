"""Profile client construction in fresh processes; never send API requests."""

import argparse
import cProfile
from datetime import datetime
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = ("direct", "adapter", "akasha")


def child(args):
    sys.path.insert(0, str(ROOT))
    os.environ["GEMINI_API_KEY"] = "startup-diagnostic-placeholder"
    os.environ.pop("GOOGLE_API_KEY", None)
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT",
                 "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ.pop(name, None)
    prefix = args.output / f"{args.case}-{args.run}"
    phases = []
    stopped = threading.Event()
    main_thread = threading.get_ident()
    active_phase = "startup"

    def measure(name, function):
        nonlocal active_phase
        active_phase = name
        print(f"START {name}", flush=True)
        before = time.perf_counter()
        modules_before = set(sys.modules)
        value = function()
        elapsed = time.perf_counter() - before
        phases.append({"name": name, "seconds": elapsed,
                       "new_modules": len(set(sys.modules) - modules_before)})
        print(f"END {name}: {elapsed:.3f}s", flush=True)
        return value

    profiler = cProfile.Profile() if args.profile else None
    tracer = None
    if args.trace_imports:
        from scripts._startup_import_trace import ImportTrace
        tracer = ImportTrace(lambda: active_phase)
    with prefix.with_suffix(".stacks.txt").open("w", encoding="utf-8") as stacks:
        def capture_stacks():
            while not stopped.wait(args.stack_after):
                frame = sys._current_frames().get(main_thread)
                if frame is not None:
                    stacks.write(f"Phase: {active_phase}\n")
                    traceback.print_stack(frame, file=stacks)
                    stacks.flush()

        sampler = threading.Thread(target=capture_stacks, daemon=True)
        sampler.start()
        if profiler:
            profiler.enable()
        if tracer:
            tracer.__enter__()
        try:
            if args.case == "direct":
                module = measure("google_genai_import", lambda: __import__("google.genai", fromlist=["Client"]))
                construct = lambda: module.Client(api_key=os.environ["GEMINI_API_KEY"])
            elif args.case == "adapter":
                module = measure("langchain_google_import", lambda: __import__("langchain_google_genai", fromlist=["ChatGoogleGenerativeAI"]))
                construct = lambda: module.ChatGoogleGenerativeAI(model=args.model, api_key=os.environ["GEMINI_API_KEY"])
            elif args.case == "core":
                module = measure("langchain_core_llm_import", lambda: __import__(
                    "langchain_core.language_models.fake", fromlist=["FakeListLLM"]))
                construct = lambda: module.FakeListLLM(responses=["offline diagnostic answer"])
            else:
                module = measure("akasha_package_import", lambda: __import__("akasha"))
                ask = measure("akasha_ask_import", lambda: module.ask)
                construct = lambda: ask(model=f"gemini:{args.model}")
            first = measure("first_construct", construct)
            if args.case == "core":
                answer = measure("first_local_invoke", lambda: first.invoke("diagnostic"))
                assert answer == "offline diagnostic answer"
            second = measure("second_construct", construct)
            assert first is not second
        finally:
            if tracer:
                tracer.__exit__()
            if profiler:
                profiler.disable()
            stopped.set()
            sampler.join()
    result = {"case": args.case, "run": args.run, "profile_enabled": args.profile,
              "python": sys.executable, "phases": phases}
    if tracer:
        result["import_trace"] = tracer.result()
    if profiler:
        profiler.dump_stats(str(prefix.with_suffix(".pstats")))
        functions = [{"file": key[0], "line": key[1], "function": key[2],
                      "calls": value[1], "own_s": value[2], "cumulative_s": value[3]}
                     for key, value in pstats.Stats(profiler).stats.items()]
        for metric in ("own_s", "cumulative_s"):
            result[f"top_{metric}"] = sorted(functions, key=lambda f: f[metric], reverse=True)[:30]
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    first_seconds = sum(p["seconds"] for p in phases if p["name"] != "second_construct")
    print(f"FIRST_READY {first_seconds:.3f}s; threshold={args.threshold:.3f}s", flush=True)
    return int(first_seconds > args.threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stack-after", type=float, default=5)
    parser.add_argument("--threshold", type=float, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-imports", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=CASES + ("core",), help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.stack_after <= 0 or args.threshold <= 0:
        parser.error("repeats, stack-after and threshold must be positive")
    if args.output is None:
        args.output = ROOT / "artifacts" / "startup-profile" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        return child(args)
    print(f"Artifacts: {args.output}", flush=True)
    failed = False
    runs = []
    for run in range(1, args.repeats + 1):
        # Rotate order to expose cache effects without concurrent contention.
        for case in CASES[run % 3:] + CASES[:run % 3]:
            command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                       "--run", str(run), "--model", args.model, "--output", str(args.output),
                       "--stack-after", str(args.stack_after), "--threshold", str(args.threshold)]
            if args.profile:
                command.append("--profile")
            if args.trace_imports:
                command.append("--trace-imports")
            print(f"CASE {case}, run {run}", flush=True)
            log_path = args.output / f"{case}-{run}.log"
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=120)
                    failed |= result.returncode != 0
                    runs.append({"case": case, "run": run, "returncode": result.returncode})
                except subprocess.TimeoutExpired:
                    log.write("Startup exceeded the 120-second deadline.\n")
                    failed = True
                    runs.append({"case": case, "run": run, "timeout": True})
            print(log_path.read_text(encoding="utf-8"), flush=True)
            print(f"RESULT {runs[-1]}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
