"""Build retrievers and call the current search_docs and retri_docs APIs."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import akasha.helper as ah
    import akasha.utils.db as adb
    from akasha.utils.search.retrievers.base import get_retrivers
    from akasha.utils.search.search_doc import search_docs, retri_docs
    from examples._common import require_documents, copy_documents
    with workspace(args, "retrieval"):
        source = copy_documents()
        embeddings = ah.handle_embeddings(args.embeddings, env_file=args.env_file)
        db, ignored = adb.process_db(source, embeddings, chunk_size=500)
        require_documents(db, ignored)
        retrievers = get_retrivers(db=db, embeddings=embeddings, search_type="knn")
        query = "How is predictive maintenance used?"
        docs, length, tokens = search_docs(retrievers, query, model=args.model,
                                           search_type="knn", language="en", max_input_tokens=3000)
        print("Documents:", [doc.page_content for doc in docs])
        print("Words:", length, "Tokens:", tokens)
        ranked, scores = retrievers[0].get_relevant_documents_and_scores(query)
        print("Scores:", scores)
        # retri_docs takes retrievers first; there is no leading db argument.
        selected = retri_docs(retrievers, query, search_type="knn", topK=3)
        print("Top results:", [doc.page_content for doc in selected])


if __name__ == "__main__":
    main()
