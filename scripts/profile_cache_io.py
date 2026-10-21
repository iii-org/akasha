"""Measure small-file operations without importing Akasha or model SDKs."""

import argparse
from datetime import datetime
import json
import importlib.util
import marshal
import os
from pathlib import Path
import struct
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bytecode", action="store_true", help="Write unique valid Python bytecode instead of repeated text")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "artifacts" / "startup-profile" / datetime.now().strftime("io-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    payload = b"startup filesystem diagnostic\n" * 600
    results = []
    for extension in ("pyc", "bin"):
        directory = output / "data" / extension
        directory.mkdir(parents=True)
        for run in (1, 2):
            totals = dict(open=0.0, write=0.0, close=0.0, replace=0.0)
            maximum = dict(open=0.0, write=0.0, close=0.0, replace=0.0)
            for index in range(args.count):
                if args.bytecode:
                    source = f"values = {list(range(index, index + 1024))!r}\n"
                    code = compile(source, f"module-{index}.py", "exec")
                    payload = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, 0, len(source)) + marshal.dumps(code)
                target = directory / f"module-{index}.{extension}"
                temporary = directory / f"module-{index}.{extension}.{id(payload)}"
                before = time.perf_counter()
                fd = os.open(temporary, os.O_EXCL | os.O_CREAT | os.O_WRONLY, 0o666)
                elapsed = time.perf_counter() - before
                totals["open"] += elapsed
                maximum["open"] = max(maximum["open"], elapsed)
                file = os.fdopen(fd, "wb", buffering=0)
                try:
                    before = time.perf_counter()
                    file.write(payload)
                    elapsed = time.perf_counter() - before
                    totals["write"] += elapsed
                    maximum["write"] = max(maximum["write"], elapsed)
                finally:
                    before = time.perf_counter()
                    file.close()
                    elapsed = time.perf_counter() - before
                    totals["close"] += elapsed
                    maximum["close"] = max(maximum["close"], elapsed)
                before = time.perf_counter()
                os.replace(temporary, target)
                elapsed = time.perf_counter() - before
                totals["replace"] += elapsed
                maximum["replace"] = max(maximum["replace"], elapsed)
            result = {"extension": extension, "run": run, "count": args.count,
                      "unique_bytecode": args.bytecode,
                      "bytes_per_file": len(payload), "total_s": totals,
                      "average_ms": {name: seconds * 1000 / args.count for name, seconds in totals.items()},
                      "maximum_ms": {name: seconds * 1000 for name, seconds in maximum.items()}}
            results.append(result)
            print(json.dumps(result), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
