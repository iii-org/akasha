from langchain_chroma import Chroma
import logging
from pathlib import Path
from typing import Union, Callable
from langchain_core.embeddings import Embeddings
from akasha.utils.db.db_structure import get_storage_directory
from akasha.helper.base import get_embedding_type_and_name
import shutil


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
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)
    try:
        embed_type, embed_name = get_embedding_type_and_name(embeddings)
        storage_directory = get_storage_directory(directory_path, chunk_size,
                                                  embed_type, embed_name)

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
    storage_directory = get_storage_directory(doc_path, chunk_size, embed_type,
                                              embed_name)

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

    docsearch._client._system.stop()
    docsearch = None
    del docsearch

    return len(ids_to_delete)


def delete_documents_from_chroma_by_file_name(chroma: Chroma,
                                              file_name: str) -> int:
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
