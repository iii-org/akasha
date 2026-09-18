"""Call a running Akasha HTTP API with real environment settings."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

import os

def build_payload(args):
    keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_BASE_URL", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
            "SERPER_API_KEY", "BRAVE_API_KEY")
    common = {"model": args.model,
              "env_config": {key: os.environ[key] for key in keys if os.environ.get(key)}}
    if args.action == "summary":
        return {**common, "content": (DATA / "industry.txt").read_text(encoding="utf-8"),
                "summary_type": "map_reduce", "summary_len": 150}
    if args.action == "rag":
        return {**common, "data_source": args.data_source or str(DATA),
                "prompt": "How can predictive maintenance reduce downtime?",
                "embedding_model": args.embeddings, "search_type": "knn"}
    if args.action == "websearch":
        return {**common, "prompt": "What is Industry 4.0?",
                "search_engine": args.engine, "search_num": 3}
    return {**common, "prompt": "What is Industry 4.0?",
            "info": (DATA / "industry.txt").read_text(encoding="utf-8")}

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--action", choices=["ask", "rag", "summary", "websearch"], default="ask")
    cli.add_argument("--base-url", help="API_BASE_URL or API_HOST:API_PORT (default localhost:8000)")
    cli.add_argument("--data-source", help="RAG path as seen by the API server")
    cli.add_argument("--engine", choices=["wiki", "serper", "brave"], default="wiki")
    args = configure(cli, argv)
    import requests
    base = args.base_url or os.getenv("API_BASE_URL")
    if not base:
        base = os.getenv("API_HOST", "http://127.0.0.1").rstrip("/") + ":" + os.getenv("API_PORT", "8000")
    route = "RAG" if args.action == "rag" else args.action
    response = requests.post(base.rstrip("/") + "/" + route,
                             json=build_payload(args), timeout=180)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict) and result.get("status") not in (None, "success"):
        raise RuntimeError(f"Akasha API returned status: {result.get('status')}")
    print(result)


if __name__ == "__main__":
    main()
