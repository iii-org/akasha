"""Run the bundled python-repl-skill with automatic Agent progress."""
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
    with workspace(args, "repl_app"):
        agent = akasha.agents(model=args.model, skills=[str(BASE_DIR / "python-repl-skill")],
                              env_file=args.env_file, stream=True,
                              thinking=args.thinking, verbose=True, keep_logs=True)
        response = agent("Use the python-repl-skill. Create values=[2,4,6,8] and total=sum(values) in one execution. In a separate execution, reuse those variables to report the average.")
        for _event in response:
            pass  # verbose already prints progress, tools, thinking and answer.
        agent.save_logs("repl_app.json")


if __name__ == "__main__":
    main()
