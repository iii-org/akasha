"""Measure real uv installation with/without bytecode compilation and live Ask."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def clean_environment():
    env = dict(os.environ, LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
    for name in ("PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE", "UV_COMPILE_BYTECODE",
                 "AKASHA_STARTUP_CACHE_OPS", "VIRTUAL_ENV", "PYTHONPATH",
                 "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
                 "GOOGLE_APPLICATION_CREDENTIALS"):
        env.pop(name, None)
    return env


def inventory(path):
    if os.name == "nt":
        path = Path("\\\\?\\" + str(path.resolve()))
    files = list(path.rglob("*.pyc"))
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def command(args, env, output, timeout=1200, json_output=False):
    start = time.perf_counter()
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    log = result.stdout if json_output else result.stdout + result.stderr
    for name, value in env.items():
        if value and any(word in name.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            log = log.replace(value, "<REDACTED>")
    output.write_text(log, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed; exit={result.returncode}; log={output}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--environments", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prepared-assets", type=Path,
                        help="Reuse previously prepared wheel, lock, and primed uv cache")
    args = parser.parse_args()
    output = args.output.resolve()
    assets = args.prepared_assets.resolve() if args.prepared_assets else output
    output.mkdir(parents=True, exist_ok=True)
    envroot = args.environments.resolve()
    wheels = list((assets / "wheel").glob("*.whl"))
    lock = assets / "pinned-light.txt"
    if len(wheels) != 1 or not lock.exists() or envroot.exists() or args.repeats < 1:
        parser.error("Require one prepared wheel, pinned-light.txt, and a new environment directory")
    envroot.mkdir(parents=True)
    env = clean_environment()

    def create(path, label):
        command(["uv", "venv", str(path), "--python", sys.executable, "--no-config"],
                env, output / f"{label}-venv.log")
        return path / "Scripts/python.exe"

    def install(python, label, compile_bytecode, offline):
        cmd = ["uv", "pip", "install", "--python", str(python), "-r", str(lock),
               str(wheels[0]), "--no-deps", "--no-config", "--no-progress"]
        if compile_bytecode:
            cmd.append("--compile-bytecode")
        if offline:
            cmd.append("--offline")
        return command(cmd, env, output / f"{label}-install.log")

    # Prime the shared uv distribution cache before measuring either condition.
    if args.prepared_assets:
        prep_seconds = json.loads((assets / "setup.json").read_text())["cache_prime_install_seconds"]
    else:
        bootstrap = create(envroot / "prep", "prep")
        prep_seconds = install(bootstrap, "prep", False, False)
    (output / "setup.json").write_text(json.dumps({"cache_prime_install_seconds": prep_seconds}, indent=2), encoding="utf-8")
    print("SETUP cache priming install", round(prep_seconds, 3), flush=True)
    records = []
    for run in range(1, args.repeats + 1):
        order = ("normal", "compiled") if run % 2 else ("compiled", "normal")
        for mode in order:
            label = f"{mode}-{run}"
            path = envroot / ("n" if mode == "normal" else "c") / str(run)
            python = create(path, label)
            elapsed = install(python, label, mode == "compiled", True)
            packages = path / "Lib/site-packages"
            before = inventory(packages)
            manifest_path = output / f"{label}-packages.json"
            command(["uv", "pip", "list", "--python", str(python), "--format", "json", "--no-config"],
                    env, manifest_path, json_output=True)
            manifest = json.loads(manifest_path.read_text())
            if any(p["name"].lower() in ("torch", "transformers", "faiss-cpu", "mlflow") for p in manifest):
                raise RuntimeError("Full-only package found in light installation")
            row = {"mode": mode, "run": run, "install_seconds": elapsed,
                   "packages": manifest, "cache_after_install": before, "startup": {}}
            print("INSTALL", label, round(elapsed, 3), "pyc", before["files"], flush=True)
            for condition in ("first", "second"):
                result_path = output / f"{label}-{condition}.json"
                cmd = [str(python), str(ROOT / "scripts/benchmark_feature_startup.py"), "--child",
                       "--source", str(packages), "--output", str(result_path),
                       "--env-file", str(ROOT / "tests/.env"), "--live"]
                wall = command(cmd, env, output / f"{label}-{condition}.log", timeout=180)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not result["phases"]["response_exact"]:
                    raise RuntimeError("Unexpected smoke response")
                after = inventory(packages)
                result.update(process_wall_seconds=wall, new_pyc_files=after["files"] - before["files"])
                row["startup"][condition] = result
                before = after
                print("STARTUP", label, condition, result["phases"], "new_pyc", result["new_pyc_files"], flush=True)
            records.append(row)
            (output / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    manifests = [sorted((p["name"], p["version"]) for p in r["packages"]) for r in records]
    if any(m != manifests[0] for m in manifests[1:]):
        raise RuntimeError("Installed package versions differ")
    summary = {"cache_prime_install_seconds": prep_seconds, "same_package_versions": True,
               "package_count": len(manifests[0]), "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
               "note": "Offline install from primed uv cache; normal pycache locations; shared existing Python stdlib cache. Not machine cold."}
    for mode in ("normal", "compiled"):
        group = [r for r in records if r["mode"] == mode]
        summary[mode] = {"install_seconds": [r["install_seconds"] for r in group],
                         "install_median_seconds": statistics.median(r["install_seconds"] for r in group),
                         "cache_after_install": [r["cache_after_install"] for r in group]}
        for condition in ("first", "second"):
            summary[mode][condition] = {metric: statistics.median(r["startup"][condition]["phases"][metric] for r in group)
                                       for metric in group[0]["startup"][condition]["phases"] if metric.endswith("_seconds")}
            summary[mode][condition]["new_pyc_files"] = [r["startup"][condition]["new_pyc_files"] for r in group]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
