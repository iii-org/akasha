from typing import Union, List, Tuple, Callable, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from akasha.utils.db.db_structure import dbs, get_storage_directory, is_url
from akasha.utils.db.extract_db import extract_db_by_file
from akasha.utils.db.create_db import (
    create_directory_db,
    create_single_file_db,
    create_webpage_db,
)
from akasha.helper import separate_name
from akasha.helper.handle_objects import handle_embeddings_and_name
from langchain_chroma import Chroma
from chromadb.config import Settings
import logging
import gc
from collections import defaultdict
from tqdm import tqdm


def process_db(
    data_source: Union[List[Union[str, Path]], Union[Path, str]],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
    verbose: bool = False,
    env_file: str = "",
) -> Tuple[dbs, List[str]]:
    """create and load dbs object from data_source

    Args:
        data_source (Union[List[Union[str,Path]], Union[Path,str]]): _description_

    Returns:
        Tuple[dbs, List[str]]: return dbs object and list of ignored files
    """
    ignored_files = []
    embeddings, embeddings_name = handle_embeddings_and_name(
        embeddings, False, env_file
    )
    tot_db = dbs()

    direct_list = []
    files_dict = defaultdict(list)
    suc_files = []
    link_list = []
    file_count = 0
    if not isinstance(data_source, list):
        data_source = [data_source]

    for data_path in data_source:
        if isinstance(data_path, str):
            if data_path == "":
                continue
            if is_url(data_path):
                link_list.append(data_path)
                continue
            data_path = Path(data_path)

        if not data_path.exists():
            logging.warning(f"File {data_path} not found")
            print(f"File {data_path} not found")
            ignored_files.append(data_path)
            continue

        ### divide data_path into dir, files and links ###
        if data_path.is_dir():
            direct_list.append(data_path)

        else:  # if files
            files_dict[data_path.parent.__str__()].append(data_path)
            file_count += 1

    ### load dbs object based on directories ###
    for data_path in direct_list:
        try:
            is_suc, cur_ignores = create_directory_db(
                data_path,
                embeddings,
                chunk_size,
                env_file=env_file,
            )
            ignored_files.extend(cur_ignores)

            if is_suc:
                new_dbs = load_directory_db(
                    data_path,
                    embeddings,
                    chunk_size,
                )
                tot_db.merge(new_dbs)
        except Exception as e:
            logging.warning(f"Error loading directory {data_path}: {e}")
            print(f"Error loading directory {data_path}: {e}")
            continue

    ### load dbs object based on files ###
    for parent_dir, file_list in files_dict.items():
        progress = tqdm(total=len(file_list), desc=f"db {parent_dir}")

        for data_path in file_list:
            progress.update(1)
            try:
                is_suc = create_single_file_db(
                    data_path, embeddings, chunk_size, env_file=env_file
                )

                if is_suc:
                    suc_files.append(data_path)
                else:
                    ignored_files.append(data_path)
            except Exception as e:
                logging.warning(f"Error loading file {data_path}: {e}")
                print(f"Error loading file {data_path}: {e}")
                ignored_files.append(data_path)
                continue
        progress.close()

    if len(suc_files) > 0:
        new_dbs = load_files_db(suc_files, embeddings, chunk_size)
        tot_db.merge(new_dbs)

    ### load dbs object based on links ###
    if len(link_list) > 0:
        progress = tqdm(total=len(link_list), desc="db http")
        for url in link_list:
            progress.update(1)
            try:
                is_suc = create_webpage_db(
                    url,
                    embeddings,
                    chunk_size,
                    env_file=env_file,
                )

                if is_suc:
                    new_dbs = load_directory_db(
                        url,
                        embeddings,
                        chunk_size,
                    )
                    tot_db.merge(new_dbs)
                else:
                    ignored_files.append(url)
            except Exception as e:
                logging.warning(f"Error loading directory {data_path}: {e}")
                print(f"Error loading directory {data_path}: {e}")
                continue
        progress.close()

    ### print the information ###
    _display_db_num(len(direct_list), file_count, len(link_list), verbose)

    return tot_db, ignored_files


def load_db_by_chroma_name(
    chroma_name_list: Union[List[Union[str, Path]], Union[Path, str]],
) -> Tuple[dbs, List[str]]:
    """load dbs object from chroma db name

    Args:
        chroma_name_list (List[Union[str, Path]]): _description_

    Returns:
        Tuple[dbs, List[str]]: return dbs object and list of ignored files
    """

    if not isinstance(chroma_name_list, list):
        chroma_name_list = [chroma_name_list]

    ignored_files = []
    tot_db = dbs()
    for chroma_name in chroma_name_list:
        if isinstance(chroma_name, Path):
            chroma_name = chroma_name.__str__()

        if not Path(chroma_name).exists():
            logging.warning(f"File {chroma_name} not found")
            print(f"File {chroma_name} not found")
            ignored_files.append(chroma_name)
            continue

        client_settings = Settings(
            is_persistent=True,
            persist_directory=chroma_name,
            anonymized_telemetry=False,
        )
        docsearch = Chroma(
            persist_directory=chroma_name, client_settings=client_settings
        )
        new_dbs = dbs(docsearch)

        if len(new_dbs.get_ids()) == 0:
            logging.warning(f"No vectors found in the chromadb directory {chroma_name}")
            print(f"No vectors found in the chromadb directory {chroma_name}")
            ignored_files.append(chroma_name)
            continue
        else:
            tot_db.merge(new_dbs)

        del docsearch, new_dbs
        gc.collect()

    return tot_db, ignored_files


def load_directory_db(
    directory_path: Union[str, Path],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
    env_file: Optional[str] = "",
) -> dbs:
    ### get the chromadb directory name###
    if not isinstance(embeddings, str):
        embeddings, embeddings_name = handle_embeddings_and_name(
            embeddings, False, env_file
        )
    else:
        embeddings_name = embeddings
    embed_type, embed_name = separate_name(embeddings_name)

    storage_directory = get_storage_directory(
        directory_path, chunk_size, embed_type, embed_name
    )

    client_settings = Settings(
        is_persistent=True,
        persist_directory=storage_directory,
        anonymized_telemetry=False,
    )
    docsearch = Chroma(
        persist_directory=storage_directory, client_settings=client_settings
    )
    tot_dbs = dbs(docsearch)

    del docsearch
    gc.collect()

    if len(tot_dbs.get_ids()) == 0:
        raise ValueError(
            f"No vectors found in the chromadb directory {storage_directory}"
        )

    return tot_dbs


def load_files_db(
    file_path_list: Union[List[str], List[Path]],
    embeddings: Union[str, Embeddings, Callable],
    chunk_size: int,
    env_file: Optional[str] = "",
) -> dbs:
    dir_set = set()
    file_name_dict = defaultdict(list)
    tot_dbs = dbs()
    if not isinstance(embeddings, str):
        embeddings, embeddings_name = handle_embeddings_and_name(
            embeddings, False, env_file
        )
    else:
        embeddings_name = embeddings
    embed_type, embed_name = separate_name(embeddings_name)

    ### get the chromadb directory names and file names###
    for file_path in file_path_list:
        if not Path(file_path).exists():
            logging.warning(f"File {file_path} not found")
            print(f"File {file_path} not found")
            continue

        cur_file_name = Path(file_path).name
        cur_dir = Path(file_path).parent

        storage_directory = get_storage_directory(
            cur_dir, chunk_size, embed_type, embed_name
        )
        dir_set.add(storage_directory)
        file_name_dict[storage_directory].append(cur_file_name)

    for st_dir in dir_set:
        try:
            client_settings = Settings(
                is_persistent=True,
                persist_directory=st_dir,
                anonymized_telemetry=False,
            )
            docsearch = Chroma(
                persist_directory=st_dir, client_settings=client_settings
            )
            new_dbs = dbs(docsearch)
        except Exception as e:
            logging.warning(f"Error loading chromadb directory {st_dir}: {e}")
            print(f"Error loading chromadb directory {st_dir}: {e}")
            continue

        new_dbs = extract_db_by_file(new_dbs, file_name_dict[st_dir])
        tot_dbs.merge(new_dbs)

        del docsearch, new_dbs
        gc.collect()

    if len(tot_dbs.get_ids()) == 0:
        raise ValueError(
            f"No vectors found in the chromadb directory {storage_directory}"
        )

    return tot_dbs


def _display_db_num(dir_num: int, file_num: int, link_num: int, verbose: bool):
    if not verbose:
        return

    info = "\nload "
    if dir_num > 0:
        info += str(dir_num)
        if dir_num > 1:
            info += " directories, "
        else:
            info += " directory, "
    if file_num > 0:
        info += str(file_num)
        if file_num > 1:
            info += " files, "
        else:
            info += " file, "
    if link_num > 0:
        info += str(link_num)
        if link_num > 1:
            info += " links, "
        else:
            info += " link, "

    print(info + "\n\n")
