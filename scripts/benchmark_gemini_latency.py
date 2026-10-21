"""Bounded live comparison with metadata-only Gemini diagnostics.

Uses this checkout and the invoking interpreter's dependencies. No package or
environment installation is performed. Run with --live and an explicit env file.
"""

from __future__ import annotations

import argparse
import contextvars
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import logging
from pathlib import Path
import statistics
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRACE = contextvars.ContextVar("gemini_benchmark_trace", default=None)


def enum_value(value):
    return getattr(value, "name", value) if value is not None else None


def provider_summary(response):
    candidates = []
    for candidate in response.candidates or []:
        parts = candidate.content.parts if candidate.content else []
        candidates.append({
            "finish_reason": enum_value(candidate.finish_reason),
            "visible_chars": sum(len(p.text or "") for p in parts or [] if not p.thought),
            "thinking_chars": sum(len(p.text or "") for p in parts or [] if p.thought),
            "tool_calls": sum(p.function_call is not None for p in parts or []),
            "parts": len(parts or []),
        })
    usage = response.usage_metadata
    return {
        "candidates": candidates,
        "block_reason": enum_value(getattr(response.prompt_feedback, "block_reason", None)),
        "usage": {name: getattr(usage, name, None) for name in (
            "prompt_token_count", "candidates_token_count", "thoughts_token_count",
            "total_token_count",
        )} if usage else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--modes", nargs="+", choices=["default", "local", "direct"],
                        default=["default", "local", "direct"])
    parser.add_argument("--max-output-tokens", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-generations", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    minimum = len(args.modes) * (1 + sum(args.concurrency) * args.repeats)
    if not args.live:
        parser.error("--live is required to make external model requests")
    if (not args.concurrency or min(args.concurrency) < 1 or args.repeats < 1
            or args.timeout <= 0 or args.max_output_tokens < 1
            or len(set(args.modes)) != len(args.modes) or minimum > args.max_generations):
        parser.error("invalid limits or planned requests exceed --max-generations")
    args.output.mkdir(parents=True, exist_ok=False)
    # Old call_model logs whole exception tracebacks, which may contain URLs.
    # This harness emits only the explicit metadata allowlist below.
    logging.disable(logging.CRITICAL)

    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    import akasha
    from akasha.helper.run_llm import content_to_text
    from akasha.helper.token_counter import myTokenizer
    from google.genai import models, types
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    token_dispatch = myTokenizer.compute_tokens_gemini

    def token_count(text, model_name, **kwargs):
        event = {"operation": "token_budget"}
        TRACE.get()["events"].append(event)
        start = time.perf_counter()
        try:
            if TRACE.get()["mode"] == "local":
                return len(encoding.encode(text, disallowed_special=()))
            return token_dispatch(text, model_name, **kwargs)
        finally:
            event["seconds"] = round(time.perf_counter() - start, 4)

    clients = {mode: akasha.ask(
        model=f"gemini:{args.model}", temperature=0, max_input_tokens=1048576,
        max_output_tokens=args.max_output_tokens, system_prompt="Reply with only OK.",
        prompt_format_type="chat_gpt", keep_logs=False, verbose=False,
    ) for mode in args.modes}
    for qa in clients.values():
        qa.model_obj.timeout = args.timeout
        qa.model_obj.max_retries = 0
    model_class = type(next(iter(clients.values())).model_obj).__mro__[1]
    original_invoke = model_class.invoke
    original_generate = models.Models.generate_content
    original_count = models.Models.count_tokens
    budget_lock = threading.Lock()
    generations = 0

    def generate(self, *pos, **kwargs):
        nonlocal generations
        with budget_lock:
            if generations >= args.max_generations:
                raise RuntimeError("benchmark generation budget exhausted")
            generations += 1
        event = {"operation": "generate"}
        TRACE.get()["events"].append(event)
        start = time.perf_counter()
        try:
            response = original_generate(self, *pos, **kwargs)
            event.update(provider_summary(response))
            return response
        except Exception as exc:
            event["error_type"] = type(exc).__name__
            return_code = getattr(exc, "code", None)
            if isinstance(return_code, (int, str)):
                event["error_code"] = return_code
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - start, 4)

    def count(self, *pos, **kwargs):
        event = {"operation": "count_tokens"}
        TRACE.get()["events"].append(event)
        start = time.perf_counter()
        # The old helper has no deadline; constrain this diagnostic run only.
        kwargs["config"] = types.CountTokensConfig(http_options=types.HttpOptions(
            timeout=int(args.timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        try:
            response = original_count(self, *pos, **kwargs)
            event["tokens"] = response.total_tokens
            return response
        except Exception as exc:
            event["error_type"] = type(exc).__name__
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - start, 4)

    def invoke(self, *pos, **kwargs):
        response = original_invoke(self, *pos, **kwargs)
        TRACE.get()["events"].append({
            "operation": "normalized_response",
            "akasha_visible_chars": len(content_to_text(response.content)),
            "langchain_visible_chars": len(response.text),
            "content_types": [block.get("type", "unknown")
                              if isinstance(block, dict) else type(block).__name__
                              for block in response.content]
                             if isinstance(response.content, list) else ["str"],
        })
        return response

    def send(mode, concurrency, repeat, warmup=False, barrier=None):
        record = {"mode": mode, "concurrency": concurrency, "repeat": repeat,
                  "warmup": warmup, "events": []}
        trace_token = TRACE.set(record)
        if barrier:
            barrier.wait()
        start = time.perf_counter()
        try:
            qa = copy.copy(clients[mode])
            qa.logs, qa.timestamp_list, qa.docs = {}, [], []
            if mode == "direct":
                answer = qa.model_obj.invoke([
                    ("system", "Reply with only OK."),
                    ("human", "User question: Reply OK."),
                ]).text
            else:
                answer = qa("Reply OK.", history_messages=[])
            record.update(success=bool(answer.strip()), answer_ok=answer.strip() == "OK")
        except Exception as exc:
            record.update(success=False, error_type=type(exc).__name__)
        finally:
            record["seconds"] = round(time.perf_counter() - start, 4)
            TRACE.reset(trace_token)
        return record

    manifest = {
        "source": str(Path(akasha.__file__).resolve()), "python": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in (
            "akasha-terminal", "google-genai", "langchain-google-genai", "tiktoken",
        )},
        "api_backend": "vertex_ai" if next(iter(clients.values())).model_obj.client.vertexai
                       else "gemini_developer_api",
        "model": args.model, "max_output_tokens": args.max_output_tokens,
        "modes": args.modes,
        "concurrency": args.concurrency, "repeats": args.repeats,
        "max_generations": args.max_generations, "timeout": args.timeout,
        "transport": "shared synchronous LangChain/SDK; no REST",
        "notes": ["Local mode changes only Gemini token counting.",
                  "Default mode runs the current checkout without token overrides.",
                  "Provider text, prompts, reasoning and exception messages are omitted.",
                  "Count timeout is bounded for this experiment in all remote-count cases."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    try:
        with patch.object(myTokenizer, "compute_tokens_gemini", staticmethod(token_count)), \
                patch.object(models.Models, "generate_content", generate), \
                patch.object(models.Models, "count_tokens", count), \
                patch.object(model_class, "invoke", invoke), \
                (args.output / "trials.jsonl").open("w", encoding="utf-8") as output:
            def persist(record):
                records.append(record)
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(json.dumps({k: v for k, v in record.items() if k != "events"}), flush=True)

            for mode in clients:
                persist(send(mode, 1, 0, warmup=True))
            for repeat in range(args.repeats):
                for index, concurrency in enumerate(args.concurrency):
                    modes = list(clients)
                    shift = (repeat + index) % len(modes)
                    modes = modes[shift:] + modes[:shift]
                    for mode in modes:
                        barrier = threading.Barrier(concurrency)
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            futures = [pool.submit(send, mode, concurrency, repeat + 1,
                                                   barrier=barrier) for _ in range(concurrency)]
                            for future in futures:
                                persist(future.result())
    finally:
        for qa in clients.values():
            qa.model_obj.client.close()
    groups = []
    for concurrency in args.concurrency:
        for mode in clients:
            selected = [r for r in records if not r["warmup"]
                        and r["mode"] == mode and r["concurrency"] == concurrency]
            good = [r["seconds"] for r in selected if r["success"]]
            events = [e for r in selected for e in r["events"]]
            groups.append({
                "mode": mode, "concurrency": concurrency, "requests": len(selected),
                "successes": len(good),
                "mean_seconds": round(statistics.mean(good), 4) if good else None,
                "median_seconds": round(statistics.median(good), 4) if good else None,
                "count_calls": sum(e["operation"] == "count_tokens" for e in events),
                "generate_calls": sum(e["operation"] == "generate" for e in events),
            })
    summary = {"generations": generations, "groups": groups}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
