"""A local inventory tool demonstrates progress independently from thinking."""
from pathlib import Path
import sys
import dotenv

dotenv.load_dotenv()

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--stream", action="store_true")
    cli.add_argument("--thinking", action="store_true")
    args = configure(cli, argv)
    import akasha
    from langchain_core.tools import tool

    @tool
    def check_inventory(product: str) -> str:
        """Look up the demo inventory of a product."""
        return f"{product}: no stock; check the delivery date."

    @tool
    def check_delivery(product: str) -> str:
        """Look up the demo delivery date of a product."""
        return f"{product}: next Friday."

    with workspace(args, "agent"):
        agent = akasha.agents(model=args.model, tools=[check_inventory, check_delivery],
                              env_file=args.env_file, verbose=True, keep_logs=True,
                              stream=args.stream, thinking=args.thinking, max_round=6)
        response = agent("Check the stock of demo-product and, if unavailable, its next delivery date.")
        if args.stream:
            for _event in response:
                pass  # verbose already displays progress, tools and the answer.
        agent.save_logs("agent.json")


if __name__ == "__main__":
    main()
