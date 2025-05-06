from typing import Callable, List, Union
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from pathlib import Path
from akasha.utils.db.db_structure import dbs, get_storage_directory
from akasha.helper.handle_objects import handle_embeddings_and_name
from akasha.helper import separate_name
import logging
from tqdm import tqdm


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
        docsearch = Chroma(persist_directory=db_path)
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
