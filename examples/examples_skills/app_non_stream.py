"""Run the bundled hello-skill with automatic Agent progress."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

BASE_DIR = Path(__file__).resolve().parent

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--thinking", action="store_true", help="Requires a model with thinking support")
    args = configure(cli, argv)
    import akasha
    with workspace(args, "app_non_stream"):
        agent = akasha.agents(model=args.model, skills=[str(BASE_DIR / "hello-skill")],
                              env_file=args.env_file, stream=False,
                              thinking=args.thinking, verbose=True, keep_logs=True)
        response = agent("Use the hello skill to greet Alice. Execute the bundled script and return its output.")
        # Non-streaming verbose mode already prints the final answer.
        agent.save_logs("app_non_stream.json")


if __name__ == "__main__":
    main()
