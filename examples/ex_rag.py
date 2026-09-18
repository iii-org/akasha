"""Retrieve bundled reference documents, then answer a question."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace, copy_documents

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--stream", action="store_true")
    args = configure(cli, argv)
    import akasha
    with workspace(args, "rag"):
        source = copy_documents()
        client = akasha.RAG(model=args.model, embeddings=args.embeddings,
                            env_file=args.env_file, search_type="knn", keep_logs=True)
        print_response(client(data_source=source, prompt="How does predictive maintenance reduce downtime?",
                              stream=args.stream))
        print("Sources:", client.reference())
        client.save_logs("rag.json")


if __name__ == "__main__":
    main()
