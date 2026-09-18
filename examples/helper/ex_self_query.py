"""Filter bundled factory reports by metadata, then retrieve and answer."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

REPORTS = DATA.parent / "reports"
REPORT_METADATA = {
    "factory_a_2024": {"factory": "A", "year": 2024},
    "factory_b_2024": {"factory": "B", "year": 2024},
    "factory_a_2023": {"factory": "A", "year": 2023},
}

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import akasha
    import akasha.helper as ah
    import akasha.utils.db as adb
    from akasha.utils.search.retrievers.base import get_retrivers
    from examples._common import require_documents, copy_documents
    with workspace(args, "self-query"):
        source = copy_documents(REPORTS, "reports")
        embeddings = ah.handle_embeddings(args.embeddings, env_file=args.env_file)
        db, ignored = adb.process_db(source, embeddings, chunk_size=500)
        require_documents(db, ignored)
        for metadata in db.metadatas:
            metadata.update(REPORT_METADATA[Path(metadata["source"]).stem])
        adb.update_db(db, source, embeddings, chunk_size=500)
        db, ignored = adb.process_db(source, embeddings, chunk_size=500)
        require_documents(db, ignored)
        model = ah.handle_model(args.model, env_file=args.env_file)
        prompt = "What were the maintenance expenses of factory A in 2024?"
        fields = [
            {"name": "factory", "description": "Factory identifier, A or B", "type": "string"},
            {"name": "year", "description": "Report calendar year", "type": "integer"},
        ]
        filtered, query, matched = ah.self_query(prompt, model, db, fields,
                                                "Annual factory maintenance expense reports")
        require_documents(filtered, [])
        print("Matched fields:", matched)
        retriever = get_retrivers(filtered, embeddings, search_type="knn")[0]
        docs, scores = retriever.get_relevant_documents_and_scores(query)
        print("Ranked sources:", [doc.metadata for doc in docs])
        print("Scores:", scores)
        client = akasha.RAG(model=model, embeddings=embeddings, search_type="knn")
        print(client(data_source=filtered, prompt=prompt))


if __name__ == "__main__":
    main()
