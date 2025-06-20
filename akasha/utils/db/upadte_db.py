from typing import Callable, List, Union
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from pathlib import Path
from akasha.utils.db.db_structure import dbs, get_storage_directory
from akasha.helper.handle_objects import handle_embeddings_and_name
from akasha.helper import separate_name, get_mac_address
import logging
from tqdm import tqdm
import datetime


def update_db(
    db: dbs,
    data_source: Union[List[Union[str, Path]], Union[Path, str]],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
    env_file: str = "",
):
    embeddings, embeddings_name = handle_embeddings_and_name(
        embeddings, False, env_file
    )
    embed_type, embed_name = separate_name(embeddings_name)
    if not isinstance(data_source, list):
        data_source = [data_source]

    ## put db.id and db.metadatas into a dict
    db_dict = {}
    for i in range(len(db.metadatas)):
        db_dict[db.ids[i]] = db.metadatas[i]

    progress = tqdm(total=len(data_source), desc="upadte db")

    for data_path in data_source:
        db_path = get_storage_directory(data_path, chunk_size, embed_type, embed_name)

        if not Path(db_path).exists():
            logging.warning(f"db path {db_path} not found")
            print(f"db path {db_path} not found")
            continue

        progress.update(1)
        cur_meta = []
        client_settings = Settings(
            is_persistent=True,
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        docsearch = Chroma(persist_directory=db_path, client_settings=client_settings)
        doc_search_data = docsearch.get(include=["metadatas"])

        ### find the ids in db_dict that are in doc_search_data["ids"]
        for id in doc_search_data["ids"]:
            if id in db_dict:
                cur_meta.append(db_dict[id])
                del db_dict[id]

        ## update the db with the new metadata
        docsearch._collection.update(ids=doc_search_data["ids"], metadatas=cur_meta)
        docsearch._client._system.stop()
        docsearch = None
        del docsearch

    return


def add_chunks(
    data: Union[List[dict], dict],
    file_name: Union[str, Path],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
    env_file: str = "",
):
    """add chunk(s) to the chromadb collection of the given file_name

    Args:
        data (Union[List[dict], dict]): data is list of dicts or a single dict with keys "text", "search_text", and "metadata". The "text" is the text to be added, "embedding" is the embedding vector, and "metadata" is a dictionary with additional information.
        file_name (Union[str, Path]): _description_
        embeddings (Union[str, Embeddings, Callable]): _description_
        chunk_size (int): _description_
    """

    if isinstance(data, dict):
        add_data = [data]
    else:
        add_data = data
    mac_address = get_mac_address()
    formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")
    embeddings, embeddings_name = handle_embeddings_and_name(
        embeddings, False, env_file
    )
    embed_type, embed_name = separate_name(embeddings_name)
    db_path = get_storage_directory(file_name, chunk_size, embed_type, embed_name)
    client_settings = Settings(
        is_persistent=True,
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    docsearch = Chroma(persist_directory=db_path, client_settings=client_settings)
    metadatas = []
    texts = []
    vectors = []

    for d in add_data:
        metadatas.append(d.get("metadata", {}))

        if "text" not in d:
            print(f"text not found in data {d}, skipping")
            continue

        if "search_text" not in d:
            vectors.append(embeddings.embed_query(d["text"]))
        else:
            vectors.append(embeddings.embed_query(d["search_text"]))

        texts.append(d["text"])

    ids = [formatted_date + "_" + str(i) + "_" + mac_address for i in range(len(texts))]
    docsearch._collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        documents=texts,
        ids=ids,
    )

    docsearch._client._system.stop()
    docsearch = None
    del docsearch

    return ids
