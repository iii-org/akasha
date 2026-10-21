"""Summarize actual import cache operations recorded by ImportTrace."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = []
    for path in sorted(args.directory.glob("akasha-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        trace = data["import_trace"]
        operations = trace["cache_operations"]
        summaries = []
        for name in ("open", "write", "close", "replace", "mkdir", "fd_wrap"):
            events = [e for e in operations if e["operation"] == name]
            values = sorted(e["seconds"] for e in events)
            summaries.append({"operation": name, "count": len(values),
                              "total_s": sum(values),
                              "median_ms": statistics.median(values) * 1000 if values else 0,
                              "maximum_s": max(values, default=0)})
        files = [f for f in trace["files"] if f["phase"] != "second_construct"]
        counts = Counter(e["operation"] for e in operations)
        writes = sum(f["writes"] for f in files)
        assert all(counts[name] == writes for name in ("open", "write", "close", "replace")), counts
        cache = sum(f["cache_s"] for f in files)
        atomic = sum(f["atomic_s"] for f in files)
        measured = sum(e["seconds"] for e in operations if e["operation"] != "mkdir")
        result = {"file": str(path), "pid": trace["process_id"], "writes": writes,
                  "first_ready_s": sum(p["seconds"] for p in data["phases"]
                                       if p["name"] != "second_construct"),
                  "cache_s": cache, "atomic_s": atomic,
                  "atomic_other_s": atomic - measured,
                  "operations": summaries,
                  "slowest": sorted(operations, key=lambda e: e["seconds"], reverse=True)[:20],
                  "write_errors": sum(len(f.get("write_errors", [])) for f in files)}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "slowest"}), flush=True)
    (args.directory / "operation-summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
