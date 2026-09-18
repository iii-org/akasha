"""Decompose a question and retrieve evidence for its subquestions."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace, copy_documents

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--stream", action="store_true", help="Set streaming on the RAG instance")
    args = configure(cli, argv)
    import akasha
    with workspace(args, "selfask"):
        source = copy_documents()
        client = akasha.RAG(model=args.model, embeddings=args.embeddings,
                            env_file=args.env_file, search_type="knn",
                            stream=args.stream, keep_logs=True)
        # selfask_RAG has no stream keyword; it uses the instance setting.
        response = client.selfask_RAG(data_source=source,
                                     prompt="How do sensors and predictive maintenance work together?")
        print_response(response)
        client.save_logs("selfask.json")


if __name__ == "__main__":
    main()
