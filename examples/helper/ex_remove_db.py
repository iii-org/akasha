"""Extract, pop and delete documents only in a private demo index."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import shutil
    import akasha.utils.db as adb
    from examples._common import require_documents
    with workspace(args, "remove-db") as output:
        # Copy inputs so this example cannot target a user's existing document index.
        source = Path("demo-documents")
        shutil.copytree(DATA, source)
        db, ignored = adb.process_db(str(source), args.embeddings, chunk_size=500,
                                     env_file=args.env_file)
        require_documents(db, ignored)
        selected_file = str(source / "maintenance.txt")
        print("By file:", adb.extract_db_by_file(db, [selected_file]).get_docs())
        print("By keyword:", adb.extract_db_by_keyword(db, ["sensors"]).get_docs())
        selected_ids = db.get_ids()[:2]
        print("By IDs:", adb.extract_db_by_ids(db, selected_ids).get_docs())
        adb.pop_db_by_ids(db, selected_ids)  # In-memory removal.
        print("Remaining in memory:", len(db.get_docs()))
        removed = adb.delete_documents_by_file(selected_file, args.embeddings, 500)
        if removed < 1:
            raise RuntimeError("The demo document was not removed from the index.")
        print("Deleted stored chunks:", removed)
        # Do not rmtree an open Chroma index: Windows may retain SQLite handles.
        # The private run directory is retained for inspection after process exit.


if __name__ == "__main__":
    main()
