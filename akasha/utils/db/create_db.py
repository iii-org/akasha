from typing import Union, List, Set, Tuple, Callable, Optional
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import logging, datetime, time, shutil
from tqdm import tqdm
from akasha.helper import get_mac_address, separate_name, handle_embeddings_and_name
from akasha.utils.db.file_loader import load_file, get_load_file_list
from akasha.utils.db.db_structure import TEXT_EXTENSIONS, get_storage_directory, FILE_LAST_CHANGE_FILE_NAME
from akasha.utils.db.db_structure import OLD_BUILT, NOT_BUILT, ALREADY_BUILT, HNSW_THRESHOLD
from akasha.utils.db.delete_db import delete_documents_from_chroma_by_file_name


def create_directory_db(directory_path: Union[str, Path],
                        embeddings: Union[str, Embeddings],
                        chunk_size: int,
                        sleep_time: int = 60,
                        env_file: str = "",
                        verbose: bool = False) -> Tuple[bool, list]:
    """If the documents vector storage not exist, create chromadb based on all .pdf files in doc_path.
        It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        directory_path (Union[str, Path]): _description_
        embeddings (Union[str, Embeddings]): _description_
        chunk_size (int): _description_
        sleep_time (int, optional): _description_. Defaults to 60.
        env_file (str, optional): _description_. Defaults to "".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[dbs, list]: _description_
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    if not directory_path.exists():
        logging.warning(f"{directory_path} does not exist.\n\n")
        return False, []

    ### get the chromadb directory name###
    embeddings, embeddings_name = handle_embeddings_and_name(
        embeddings, False, env_file)
    embed_type, embed_name = separate_name(embeddings_name)

    storage_directory = get_storage_directory(directory_path, chunk_size,
                                              embed_type, embed_name)
    files, db_path_names = [], []
    suc = 0
    m_time_list, suc_file_list = [], []
    for extension in TEXT_EXTENSIONS:
        files.extend(get_load_file_list(directory_path, extension))

    progress = tqdm(total=len(files), desc="Vec Storage")
    docsearch = Chroma(
        persist_directory=storage_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:sync_threshold": HNSW_THRESHOLD})

    for file in files:
        progress.update(1)
        whole_file_path = directory_path / file
        last_m_time = whole_file_path.stat().st_mtime

        is_doc_b = _is_doc_built(storage_directory, last_m_time, file)
        if is_doc_b == ALREADY_BUILT:
            continue
        elif is_doc_b == OLD_BUILT:
            delete_documents_from_chroma_by_file_name(docsearch, file)

        file_doc = load_file(whole_file_path, file.split(".")[-1])
        if file_doc == "" or len(file_doc) == 0:
            logging.warning(f"file {file} load failed or empty.\n\n")

        status = create_chromadb_from_file(
            file_doc,
            docsearch,
            chunk_size,
            embeddings,
            file,
            sleep_time,
        )
        if not status:
            logging.warning(f"create chromadb for {file} failed.\n\n")
            db_path_names.append(file)
        else:
            suc += 1
            suc_file_list.append(file)
            m_time_list.append(last_m_time)

    progress.close()
    id_len = len(docsearch._collection.get()['ids'])

    docsearch._client._system.stop()
    docsearch = None
    del docsearch

    ### delete the storage directory if no vector created ###
    if id_len == 0:
        logging.warning(f"can no create any vector from {directory_path}.\n\n")
        shutil.rmtree(storage_directory)
        return False, db_path_names

    #### write the last modified time of the files into json file ####
    _write_docs_built_time(storage_directory, m_time_list, suc_file_list)

    ### print the information ###
    info = "\n\nload files:" + str(len(files)) + "\n\n"
    if verbose:
        print(info)

    return True, db_path_names


def create_single_file_db(file_path: Union[str, Path],
                          embeddings: Union[str, Embeddings],
                          chunk_size: int,
                          sleep_time: int = 60,
                          env_file: str = "") -> bool:
    """create chromadb based on a single file. It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        file_path (Union[str, Path]): _description_
        embeddings (Union[str, Embeddings]): _description_
        chunk_size (int): _description_
        sleep_time (int, optional): _description_. Defaults to 60.
        env_file (str, optional): _description_. Defaults to "".

    Returns:
        bool: _description_
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        doc_path = file_path.parent
        file_name = file_path.name
    except:
        logging.warning("file path error.\n\n")
        return False

    ### get the chromadb directory name###
    embeddings, embeddings_name = handle_embeddings_and_name(
        embeddings, False, env_file)
    embed_type, embed_name = separate_name(embeddings_name)
    storage_directory = get_storage_directory(doc_path, chunk_size, embed_type,
                                              embed_name)

    last_m_time = file_path.stat().st_mtime

    is_doc_b = _is_doc_built(storage_directory, last_m_time, file_name)
    if is_doc_b == ALREADY_BUILT:
        return True, storage_directory
    elif is_doc_b == OLD_BUILT:
        delete_documents_from_chroma_by_file_name(docsearch, file_name)

    file_doc = load_file(doc_path / file_name, file_name.split(".")[-1])
    if file_doc == "" or len(file_doc) == 0:
        logging.warning("file load failed or empty.\n\n")
        return False

    docsearch = Chroma(
        persist_directory=storage_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:sync_threshold": HNSW_THRESHOLD})

    status = create_chromadb_from_file(
        file_doc,
        docsearch,
        chunk_size,
        embeddings,
        file_name,
        sleep_time,
    )

    docsearch._client._system.stop()
    docsearch = None
    del docsearch

    if not status:
        logging.warning(f"create chromadb for {file_name} failed.\n\n")

        return False

    _write_docs_built_time(storage_directory, [last_m_time], [file_name])

    return True


def create_chromadb_from_file(
    documents: List[Document],
    docsearch: Chroma,
    chunk_size: int,
    embeddings: vars,
    file_name: str,
    sleep_time: int = 60,
) -> bool:
    """load the existing chromadb of documents from storage_directory and return it, if not exist, create it.

    Args:
        documents (list): list of Documents
        docsearch (Chroma): chromadb object
        chunk_size (int): the chunk size of documents
        embeddings (vars): the embeddings used in transfer documents into vector storage
        file_name (str): the path and name of the file
        sleep_time (int, optional): waiting time if exceed api calls. Defaults to 60.
       
    Returns:
        (chromadb object): return the dbs object
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ",", ".", "。", "!"],
        chunk_size=chunk_size,
        chunk_overlap=100,
    )
    k = 0
    cum_ids = 0
    interval = 10
    mac_address = get_mac_address()
    is_added = False
    while k < len(documents):
        cur_doc = documents[k:k + interval]
        texts = text_splitter.split_documents(cur_doc)
        formatted_date = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H_%M_%S_%f")

        page_contents = [text.page_content for text in texts]

        try:
            vectors = embeddings.embed_documents(page_contents)
        except:
            time.sleep(sleep_time)
            vectors = embeddings.embed_documents(page_contents)

        if len(vectors) == 0:
            #logging.warning(f" {file_name} has empty content, ignored.\n")
            k += interval
            cum_ids += len(texts)
            continue
        docsearch._collection.add(
            embeddings=vectors, metadatas=[text.metadata for text in texts], documents=[text.page_content for text in texts]\
                , ids=[formatted_date + "_" + str(cum_ids + i) + "_" + mac_address for i in range(len(texts))]
        )
        is_added = True

        k += interval
        cum_ids += len(texts)

    if not is_added:
        logging.warning(f"\n{file_name} has empty content, ignored.\n")
        return False
    return True


def _is_doc_built(storage_directory: str, last_m_time: float,
                  file_name: str) -> str:

    storage_path = Path(storage_directory)
    if not storage_path.exists():
        return NOT_BUILT

    # Path to the JSON file
    json_file_path = storage_path / FILE_LAST_CHANGE_FILE_NAME

    # Check if the JSON file exists
    if not json_file_path.exists():
        return NOT_BUILT

    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            file_last_changed = json.load(json_file)
    except Exception as e:
        logging.warning(f"Error reading JSON file: {e}")
        return NOT_BUILT

    # Check if the file_name is in the dictionary
    if file_name not in file_last_changed:
        return NOT_BUILT

    # Check if the last modify time matches
    if file_last_changed[file_name] != last_m_time:
        return OLD_BUILT

    return ALREADY_BUILT


def _write_docs_built_time(storage_directory: str, last_m_time: List[float],
                           file_name: List[str]):

    storage_path = Path(storage_directory)

    # Path to the JSON file
    json_file_path = storage_path / FILE_LAST_CHANGE_FILE_NAME

    # Create the dictionary
    file_last_changed = {}

    # Check if the JSON file exists
    if json_file_path.exists():
        # Load the JSON file
        try:
            with open(json_file_path, 'r', encoding="utf-8") as json_file:
                file_last_changed = json.load(json_file)
        except Exception as e:
            logging.warning(f"Error reading JSON file: {e}")

    # Update the dictionary
    for f_name, m_time in zip(file_name, last_m_time):
        file_last_changed[f_name] = m_time

    # Write the dictionary to the JSON file
    try:
        with open(json_file_path, 'w', encoding="utf-8") as json_file:
            json.dump(file_last_changed,
                      json_file,
                      indent=4,
                      ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Error writing JSON file: {e}")
