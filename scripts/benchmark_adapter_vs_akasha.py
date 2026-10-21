"""Compare the same Gemini adapter directly and through Akasha, with real calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Reply with exactly: ASK_SMOKE_OK"
SYSTEM = "You are a concise assistant. Return only the requested token."


def encode(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    raise TypeError(type(value).__name__)


def child(args):
    sys.path.insert(0, str(ROOT))
    phases = {}
    start = time.perf_counter()
    if args.case == "direct":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from google.genai import types

        class DirectGemini(ChatGoogleGenerativeAI):
            # Match Akasha's adapter override, without importing Akasha.
            def _prepare_request(self, *a, **kw):
                request = super()._prepare_request(*a, **kw)
                request["config"].automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
                return request

        factory = DirectGemini
    else:
        import akasha
        factory = akasha.ask
    phases["import_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    if args.case == "direct":
        llm = factory(model=args.model, api_key=os.environ["GEMINI_API_KEY"],
                      temperature=0.0, max_output_tokens=128)
        qa = None
    else:
        qa = factory(model=f"gemini:{args.model}", env_file="", temperature=0.0,
                     max_output_tokens=128, language="ch", system_prompt=SYSTEM,
                     prompt_format_type="chat_gpt", thinking=False)
        llm = qa.model_obj
    phases["construct_seconds"] = time.perf_counter() - start
    phases["ready_seconds"] = phases["import_seconds"] + phases["construct_seconds"]

    adapter_class = type(llm)
    original_invoke = adapter_class.invoke
    original_prepare = adapter_class._prepare_request
    invokes, requests = [], []

    def timed_invoke(self, *a, **kw):
        start = time.perf_counter()
        try:
            return original_invoke(self, *a, **kw)
        finally:
            invokes.append(time.perf_counter() - start)

    def checked_prepare(self, *a, **kw):
        request = original_prepare(self, *a, **kw)
        config = request["config"]
        selected = {"model": request.get("model"), "contents": request.get("contents"),
                    "config": {name: getattr(config, name, None) for name in
                               ("system_instruction", "temperature", "max_output_tokens",
                                "thinking_config", "automatic_function_calling", "tools")}}
        payload = json.dumps(selected, default=encode, sort_keys=True)
        requests.append(hashlib.sha256(payload.encode()).hexdigest())
        return request

    adapter_class.invoke = timed_invoke
    adapter_class._prepare_request = checked_prepare
    start = time.perf_counter()
    if qa is None:
        response = llm.invoke([("system", SYSTEM), ("human", "User question: " + PROMPT)]).content
        if isinstance(response, list):
            response = "".join(x if isinstance(x, str) else x.get("text", "") for x in response)
    else:
        response = qa(PROMPT)
    phases["first_call_seconds"] = time.perf_counter() - start
    phases["adapter_invoke_seconds"] = sum(invokes)
    phases["outside_adapter_seconds"] = phases["first_call_seconds"] - sum(invokes)
    phases["first_total_seconds"] = phases["ready_seconds"] + phases["first_call_seconds"]
    if not isinstance(response, str) or response.strip() != "ASK_SMOKE_OK":
        raise RuntimeError("Unexpected smoke response")
    if len(invokes) != 1 or len(requests) != 1:
        raise RuntimeError("Expected one adapter invocation and one prepared request")
    if args.case == "direct" and any(n == "akasha" or n.startswith("akasha.") for n in sys.modules):
        raise RuntimeError("Direct test must not import Akasha")
    result = {"case": args.case, "phases": phases, "python": sys.version,
              "model": args.model, "response_exact": True, "request_hash": requests[0],
              "invocations": len(invokes), "module_count": len(sys.modules),
              "modules": sorted(sys.modules),
              "adapter_base": "langchain_google_genai.ChatGoogleGenerativeAI"}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def inventory(cache):
    if os.name == "nt":
        cache = Path("\\\\?\\" + str(cache.resolve()))
    files = list(cache.rglob("*.pyc")) if cache.exists() else []
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}


def compare(args):
    from dotenv import dotenv_values
    from importlib.metadata import version

    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    key = dotenv_values(ROOT / "tests/.env").get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Local Gemini API key is required")
    records = []
    for run in range(1, args.repeats + 1):
        order = ("direct", "akasha") if run % 2 else ("akasha", "direct")
        for case in order:
            cache = (args.output / f"cache-{case}-{run}").resolve()
            prefix = "\\\\?\\" + str(cache) if os.name == "nt" else str(cache)
            env = dict(os.environ, GEMINI_API_KEY=key, PYTHONPYCACHEPREFIX=prefix,
                       LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
            for name in ("PYTHONDONTWRITEBYTECODE", "AKASHA_STARTUP_CACHE_OPS", "GOOGLE_API_KEY",
                         "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
                         "GOOGLE_APPLICATION_CREDENTIALS"):
                env.pop(name, None)
            for condition in ("cold", "warm"):
                before = inventory(cache)
                path = args.output / f"{case}-{condition}-{run}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--child", "--case", case,
                           "--model", args.model, "--output", str(path)]
                start = time.perf_counter()
                completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
                wall = time.perf_counter() - start
                if completed.returncode:
                    # Do not relay raw provider errors that might contain credentials.
                    raise RuntimeError(f"{case}/{condition}/{run} failed; exit={completed.returncode}")
                result = json.loads(path.read_text(encoding="utf-8"))
                after = inventory(cache)
                result.update(condition=condition, run=run, cache_before=before, cache_after=after,
                              new_pyc_files=after["files"] - before["files"], process_wall_seconds=wall)
                records.append(result)
                print(case, condition, run, result["phases"], "new_pyc", result["new_pyc_files"], flush=True)
                (args.output / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if len({r["request_hash"] for r in records}) != 1:
        raise RuntimeError("Prepared request contents/configuration differ between conditions")
    summary = {"same_prepared_requests": True,
               "versions": {name: version(name) for name in ("langchain-core", "langchain-google-genai", "google-genai")},
               "note": "New processes, isolated empty or reused bytecode prefixes; OS cache not reset. Twelve real API calls for three rounds."}
    for case in ("direct", "akasha"):
        for condition in ("cold", "warm"):
            group = [r for r in records if r["case"] == case and r["condition"] == condition]
            summary[f"{case}_{condition}"] = {
                metric: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                for metric in group[0]["phases"]
                for values in [[r["phases"][metric] for r in group]]}
            summary[f"{case}_{condition}"]["new_pyc_files"] = [r["new_pyc_files"] for r in group]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=("direct", "akasha"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    child(args) if args.child else compare(args)


if __name__ == "__main__":
    main()
