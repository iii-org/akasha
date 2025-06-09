from langchain_chroma import Chroma
from chromadb.config import Settings
import logging
from pathlib import Path
from typing import Union, Callable
from langchain_core.embeddings import Embeddings
from akasha.utils.db.db_structure import (
    get_storage_directory,
    FILE_LAST_CHANGE_FILE_NAME,
)
from akasha.helper.base import get_embedding_type_and_name
import shutil
import gc
import json
from typing import List

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


def delete_documents_by_directory(
    directory_path: Union[str, Path],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
) -> int:
    """delete the documents in the chroma db by directory path

    Args:
        directory_path (Union[str, Path]): _description_
        embeddings (Union[str, Embeddings, Callable]): _description_
        chunk_size (int): _description_

    Returns:
        int: _description_
    """

    try:
        embed_type, embed_name = get_embedding_type_and_name(embeddings)
        storage_directory = get_storage_directory(
            directory_path, chunk_size, embed_type, embed_name
        )

        shutil.rmtree(Path(storage_directory))
    except Exception as e:
        logging.warning(f"Error deleting directory {directory_path}: {e}")
        print(f"Error deleting directory {directory_path}: {e}")
        return 0

    return 1


def delete_documents_by_file(
    file_path: Union[Path, str],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
) -> int:
    """delete the documents in the chroma db by file name

    Args:
        file_name (str): _description_
        embeddings (Union[str, Embeddings, Callable]): _description_
        chunk_size (int): _description_

    Returns:
        int:  the number of deleted documents
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    doc_path = file_path.parent
    file_name = file_path.name
    embed_type, embed_name = get_embedding_type_and_name(embeddings)
    storage_directory = get_storage_directory(
        doc_path, chunk_size, embed_type, embed_name
    )

    # Retrieve all documents in the collection
    client_settings = Settings(
        is_persistent=True,
        persist_directory=storage_directory,
        anonymized_telemetry=False,
    )
    docsearch = Chroma(
        persist_directory=storage_directory, client_settings=client_settings
    )
    all_docs = docsearch._collection.get()
    tot_ids_len = len(all_docs["ids"])
    # Filter documents by metadata
    ids_to_delete = [
        doci
        for doci, docm in zip(all_docs["ids"], all_docs["metadatas"])
        if file_name in docm.get("source")
    ]
    # Delete documents by IDs
    if len(ids_to_delete) > 0:
        docsearch._collection.delete(ids=ids_to_delete)

        logging.info(
            f"Deleted {len(ids_to_delete)} documents with file_name: {file_name}"
        )
    else:
        logging.warning(f"No documents found with file_name: {file_name} to delete")
        print(f"No documents found with file_name: {file_name} to delete")

    if tot_ids_len == len(ids_to_delete) or tot_ids_len == 0:
        del all_docs
        gc.collect()
        docsearch._client.delete_collection(_LANGCHAIN_DEFAULT_COLLECTION_NAME)
        del docsearch
        gc.collect()
        shutil.rmtree(Path(storage_directory))
        logging.info(f"Deleted all documents in the directory: {doc_path}")
        print(f"Deleted all documents in the directory: {doc_path}")
        return tot_ids_len
    else:
        _delete_docs_built_time(storage_directory, [file_name])

    del docsearch, all_docs
    gc.collect()
    return len(ids_to_delete)


def delete_documents_from_chroma_by_file_name(chroma: Chroma, file_name: str) -> int:
    """delete the documents in the chroma db by file name

    Args:
        chroma (Chroma): _description_
        file_name (str): _description_

    Returns:
        int: the number of deleted documents
    """
    all_docs = chroma._collection.get()
    # Filter documents by metadata
    ids_to_delete = [
        doci
        for doci, docm in zip(all_docs["ids"], all_docs["metadatas"])
        if file_name in docm.get("source")
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


def _delete_docs_built_time(storage_directory: str, file_name: List[str]):
    storage_path = Path(storage_directory)

    # Path to the JSON file
    json_file_path = storage_path / FILE_LAST_CHANGE_FILE_NAME

    # Create the dictionary
    file_last_changed = {}

    # Check if the JSON file exists
    if json_file_path.exists():
        # Load the JSON file
        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                file_last_changed = json.load(json_file)
        except Exception as e:
            logging.warning(f"Error reading JSON file: {e}")
            print(f"Error reading JSON file: {e}")
    else:
        logging.warning(f"JSON file not found: {json_file_path}")
        print(f"JSON file not found: {json_file_path}")
        return

    # Update the dictionary
    for f_name in file_name:
        file_last_changed.pop(f_name, None)

    # Write the dictionary to the JSON file
    try:
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(file_last_changed, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Error writing JSON file: {e}")
        print(f"Error writing JSON file: {e}")

    return
