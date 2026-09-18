"""Small shared setup for directly runnable examples (no work at import time)."""
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data" / "knowledge"
DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_EMBEDDINGS = "openai:text-embedding-3-small"

def parser(description):
    result = ArgumentParser(description=description)
    result.add_argument("--model", help="Model alias; defaults to AKASHA_MODEL or openai:gpt-4o-mini")
    result.add_argument("--embeddings", help="Embedding alias; defaults to AKASHA_EMBEDDINGS")
    result.add_argument("--env-file", default=os.getenv("ENV_FILE", str(ROOT / ".env")))
    result.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    return result

def configure(argument_parser, argv=None):
    args = argument_parser.parse_args(argv)
    from dotenv import load_dotenv
    env_path = Path(args.env_file).expanduser().resolve()
    # Environment variables win over dotenv; do not embed credentials in examples.
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    elif args.env_file != str(ROOT / ".env"):
        argument_parser.error(f"Environment file does not exist: {env_path}")
    # dotenv has already been merged; let Akasha read the resulting environment.
    args.env_file = ""
    args.model = args.model or os.getenv("AKASHA_MODEL", DEFAULT_MODEL)
    args.embeddings = args.embeddings or os.getenv("AKASHA_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    args.output_dir = args.output_dir.expanduser().resolve()
    for channel in (sys.stdout, sys.stderr):
        if hasattr(channel, "reconfigure"):
            channel.reconfigure(encoding="utf-8", errors="backslashreplace")
    return args

@contextmanager
def workspace(args, name):
    """Keep logs, memory and Chroma caches in a new example-owned directory."""
    previous = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=name + "-", dir=args.output_dir))
    try:
        os.chdir(target)
        yield target
    finally:
        os.chdir(previous)
        print(f"Artifacts: {target}")

def print_response(response):
    """Ask/RAG stream strings; thinking-enabled Ask and Agents stream events."""
    if isinstance(response, str):
        print(response)
        return response
    parts = []
    for event in response:
        if isinstance(event, str):
            text = event
        elif event.get("type") == "answer":
            text = event["data"]
        else:
            print(f"[{event['type']}] {event['data']}")
            continue
        parts.append(text)
        print(text, end="", flush=True)
    print()
    return "".join(parts)

def require_documents(db, ignored):
    if ignored:
        raise RuntimeError(f"Some example inputs could not be loaded: {ignored}")
    if not db.get_docs():
        raise RuntimeError("No documents were indexed. Check the embedding provider and input files.")


def copy_documents(source=DATA, name="documents"):
    """Use short relative source paths so Windows Chroma paths stay manageable."""
    import shutil
    target = Path(name + "-" + Path.cwd().name.rsplit("-", 1)[-1])
    shutil.copytree(source, target)
    return str(target)
