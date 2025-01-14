from typing import Union, List, Set, Tuple, Callable, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from akasha.utils.db.db_structure import dbs, get_storage_directory, NO_PARENT_DIR_NAME
from akasha.utils.db.extract_db import extract_db_by_file
from akasha.helper import separate_name, handle_embeddings_and_name
from langchain_chroma import Chroma
import logging
from collections import defaultdict


def load_directory_db(directory_path: Union[str, Path],
                      embeddings: Union[str, Embeddings],
                      chunk_size: int,
                      env_file: Optional[str] = "",
                      verbose: bool = False) -> dbs:

    ### get the chromadb directory name###
    if not isinstance(embeddings, str):
        embeddings, embeddings_name = handle_embeddings_and_name(
            embeddings, False, env_file)
    else:
        embeddings_name = embeddings
    embed_type, embed_name = separate_name(embeddings_name)

    storage_directory = get_storage_directory(directory_path, chunk_size,
                                              embed_type, embed_name)

    docsearch = Chroma(persist_directory=storage_directory)
    tot_dbs = dbs(docsearch)
    docsearch._client._system.stop()
    docsearch = None
    del docsearch

    if len(tot_dbs.get_ids()) == 0:
        raise ValueError(
            f"No vectors found in the chromadb directory {storage_directory}")

    return tot_dbs


def load_files_db(
    file_path_list: Union[List[str], List[Path]],
    embeddings: Union[str, Embeddings],
    chunk_size: int,
    env_file: Optional[str] = "",
) -> dbs:

    dir_set = set()
    file_name_dict = defaultdict(list)
    tot_dbs = dbs()
    if not isinstance(embeddings, str):
        embeddings, embeddings_name = handle_embeddings_and_name(
            embeddings, False, env_file)
    else:
        embeddings_name = embeddings
    embed_type, embed_name = separate_name(embeddings_name)

    ### get the chromadb directory names and file names###
    for file_path in file_path_list:
        if not Path(file_path).exists():
            logging.warning(f"File {file_path} not found")
            continue

        cur_file_name = Path(file_path).name
        cur_dir = Path(file_path).parent

        storage_directory = get_storage_directory(cur_dir, chunk_size,
                                                  embed_type, embed_name)
        dir_set.add(storage_directory)
        file_name_dict[storage_directory].append(cur_file_name)

    for st_dir in dir_set:
        try:
            docsearch = Chroma(persist_directory=st_dir)
            new_dbs = dbs(docsearch)
        except Exception as e:
            logging.warning(f"Error loading chromadb directory {st_dir}: {e}")
            continue

        new_dbs = extract_db_by_file(new_dbs, file_name_dict[st_dir])
        tot_dbs.merge(new_dbs)

        docsearch._client._system.stop()
        docsearch = None
        del docsearch, new_dbs

    if len(tot_dbs.get_ids()) == 0:
        raise ValueError(
            f"No vectors found in the chromadb directory {storage_directory}")

    return tot_dbs
