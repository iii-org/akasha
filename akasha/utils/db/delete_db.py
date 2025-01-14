from langchain_chroma import Chroma
import logging
from pathlib import Path
import time


def delete_documents_by_file_name(storage_directory: str, file_name: str):
    # Retrieve all documents in the collection

    docsearch = Chroma(persist_directory=storage_directory)
    all_docs = docsearch._collection.get()
    # Filter documents by metadata
    ids_to_delete = [
        doci for doci, docm in zip(all_docs['ids'], all_docs['metadatas'])
        if file_name in docm.get('source')
    ]
    # Delete documents by IDs
    if len(ids_to_delete) > 0:
        docsearch._collection.delete(ids=ids_to_delete)
        logging.info(
            f"Deleted {len(ids_to_delete)} documents with file_name: {file_name}"
        )
    else:
        logging.info(f"No documents found with file_name: {file_name}")

    return len(ids_to_delete)


def delete_documents_from_chroma_by_file_name(chroma: Chroma, file_name: str):

    all_docs = chroma._collection.get()
    # Filter documents by metadata
    ids_to_delete = [
        doci for doci, docm in zip(all_docs['ids'], all_docs['metadatas'])
        if file_name in docm.get('source')
    ]
    # Delete documents by IDs
    if len(ids_to_delete) > 0:
        chroma._collection.delete(ids=ids_to_delete)
        logging.info(
            f"Deleted {len(ids_to_delete)} documents with file_name: {file_name}"
        )
    else:
        logging.info(f"No documents found with file_name: {file_name}")

    return len(ids_to_delete)
