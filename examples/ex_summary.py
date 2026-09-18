"""Summarize the bundled reference documents."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--method", choices=["map_reduce", "refine"], default="map_reduce")
    args = configure(cli, argv)
    import akasha
    with workspace(args, "summary"):
        client = akasha.summary(model=args.model, sum_type=args.method,
                                sum_len=150, language="en", env_file=args.env_file,
                                max_input_tokens=8000, keep_logs=True)
        print(client(content=str(DATA)))
        client.save_logs("summary.json")


if __name__ == "__main__":
    main()
