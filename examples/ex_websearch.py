"""Answer with a configured search engine; wiki needs no separate search API key."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--engine", choices=["wiki", "serper", "brave", "tavily"], default=None)
    cli.add_argument("--stream", action="store_true")
    args = configure(cli, argv)
    import os
    import akasha
    args.engine = args.engine or os.getenv("AKASHA_SEARCH_ENGINE") or next(
        (engine for engine, key in [("brave", "BRAVE_API_KEY"), ("serper", "SERPER_API_KEY"),
                                   ("tavily", "TAVILY_API_KEY")] if os.getenv(key)), "wiki")
    if args.engine not in {"wiki", "serper", "brave", "tavily"}:
        cli.error("AKASHA_SEARCH_ENGINE must be wiki, serper, brave or tavily")
    with workspace(args, "websearch"):
        client = akasha.websearch(model=args.model, search_engine=args.engine,
                                  search_num=3, language="en", env_file=args.env_file,
                                  keep_logs=True)
        print_response(client("What is Industry 4.0?", stream=args.stream))
        client.save_logs("websearch.json")


if __name__ == "__main__":
    main()
