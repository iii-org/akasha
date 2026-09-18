"""Build and reload a Chroma index, then load source documents directly."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import akasha.utils.db as adb
    from akasha.helper import separate_name
    from examples._common import require_documents, copy_documents
    with workspace(args, "load-db"):
        source = copy_documents()
        db, ignored = adb.process_db(data_source=source, embeddings=args.embeddings,
                                      chunk_size=500, env_file=args.env_file)
        require_documents(db, ignored)
        print("Documents:", len(db.get_docs()))
        print("Embedding count:", len(db.get_embeds()))
        print("Metadata:", db.get_metadatas())
        print("IDs:", db.get_ids())
        embed_type, embed_name = separate_name(args.embeddings)
        directory = adb.get_storage_directory(source, 500, embed_type, embed_name)
        reloaded, ignored = adb.load_db_by_chroma_name(chroma_name_list=[directory])
        require_documents(reloaded, ignored)
        print("Reloaded:", len(reloaded.get_docs()))
        docs = adb.load_docs_from_info(info=source)
        print("First source document:", docs[0].page_content)


if __name__ == "__main__":
    main()
