"""Pair Kernel-File IRPs and correlate them with diagnostic cache calls."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}


def utc_ns(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    delta = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("startup", type=Path)
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    startup = json.loads(args.startup.read_text(encoding="utf-8"))
    operations = startup["import_trace"]["cache_operations"]
    prefix = next(e["path"] for e in operations if e["operation"] == "open")
    prefix = prefix.lower().split("-pyc", 1)[0].rsplit("\\", 1)[-1] + "-pyc"
    pending, names, pairs, counts = {}, {}, [], Counter()
    for _, element in ET.iterparse(args.events, events=("end",)):
        if element.tag != "{" + NS["e"] + "}Event":
            continue
        system = element.find("e:System", NS)
        event_id = int(system.findtext("e:EventID", namespaces=NS))
        execution = system.find("e:Execution", NS)
        data = {item.get("Name"): item.text for item in element.findall("e:EventData/e:Data", NS)}
        counts[event_id] += 1
        stamp = utc_ns(system.find("e:TimeCreated", NS).get("SystemTime"))
        obj = data.get("FileObject") or data.get("FileKey")
        path = data.get("FileName") or names.get(obj, "") or ""
        if obj and path:
            names[obj] = path
        irp = data.get("Irp")
        if irp and event_id != 24:
            pending[irp] = {"event_id": event_id, "start_utc_ns": stamp,
                            "path": path, "tid": int(execution.get("ThreadID")),
                            "data": data}
        elif irp and event_id == 24:
            start = pending.pop(irp, None)
            if start and prefix in start["path"].lower():
                start.update(end_utc_ns=stamp,
                             seconds=(stamp - start["start_utc_ns"]) / 1e9,
                             status=data.get("Status"))
                pairs.append(start)
        element.clear()
    comparisons = []
    for operation in sorted(operations, key=lambda e: e["seconds"], reverse=True)[:30]:
        start = operation["start_utc_ns"]
        end = start + int(operation["seconds"] * 1e9)
        suffix = operation["path"].lower().split(":\\", 1)[-1]

        def same_file(path):
            position = path.lower().find(suffix)
            if position < 0:
                return False
            tail = path[position + len(suffix):]
            return not tail or (tail.startswith(".") and tail[1:].isdigit())

        matches = [p for p in pairs if p["tid"] == operation["tid"]
                   and same_file(p["path"])
                   and start - 2_000_000 <= p["start_utc_ns"] <= end + 2_000_000
                   and p["end_utc_ns"] <= end + 2_000_000]
        comparisons.append({"operation": operation, "kernel_pairs": matches,
                            "maximum_kernel_s": max((p["seconds"] for p in matches), default=0)})
    result = {"prefix": prefix, "event_counts": dict(counts),
              "cache_irp_pairs": len(pairs), "comparisons": comparisons,
              "slowest_kernel_pairs": sorted(pairs, key=lambda p: p["seconds"], reverse=True)[:30],
              "limitations": "Process-filtered Kernel-File events, not physical DiskIO or minifilter stacks. IRP intervals do not identify a blocking driver; missing pairs can occur when completion uses another process/thread. UTC alignment has microsecond timestamp precision and a 2ms tolerance."}
    output = args.startup.parent / "etw-correlation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cache_irp_pairs": len(pairs), "event_counts": dict(counts)}))
    for comparison in comparisons[:8]:
        op = comparison["operation"]
        print(op["operation"], op["module"], round(op["seconds"], 6),
              "kernel_max_s", round(comparison["maximum_kernel_s"], 6),
              "matched_pairs", len(comparison["kernel_pairs"]))


if __name__ == "__main__":
    main()
