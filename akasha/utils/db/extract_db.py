from typing import List, Union, Set
from akasha.utils.db.db_structure import dbs
import logging
from pathlib import Path


def extract_db_by_file(db: dbs, file_name_list: List[str]) -> dbs:
    """extract db from dbs based on file_name_list

    Args:
        db (dbs): dbs object
        file_name_list (list): list of file names

    Returns:
        dbs: dbs object
    """
    ret_db = dbs()
    file_set = set()
    for file_name in file_name_list:
        file_name = file_name.replace("\\", "/")
        file_name = file_name.lstrip("./")
        file_name = file_name.split("/")[-1]
        file_set.add(file_name)

    for i in range(len(db.ids)):
        mmap = Path(db.metadatas[i]["source"]).name

        if mmap in file_set:
            ret_db.ids.append(db.ids[i])
            ret_db.embeds.append(db.embeds[i])
            ret_db.metadatas.append(db.metadatas[i])
            ret_db.docs.append(db.docs[i])
            ret_db.vis.add(db.ids[i])

    if len(ret_db.ids) == 0:
        logging.warning("No document found.\n\n")
        print("No document found.\n\n")
    return ret_db


def extract_db_by_keyword(db: dbs, keyword_list: List[str]) -> dbs:
    """extract db from dbs based on keyword_list

    Args:
        db (dbs): dbs object
        keyword_list (list): list of keywords

    Returns:
        dbs: dbs object
    """
    ret_db = dbs()
    vis_id = set()

    for keyword in keyword_list:
        for i in range(len(db.ids)):
            if db.ids[i] in vis_id:
                continue
            if keyword in db.docs[i]:
                ret_db.ids.append(db.ids[i])
                ret_db.embeds.append(db.embeds[i])
                ret_db.metadatas.append(db.metadatas[i])
                ret_db.docs.append(db.docs[i])
                ret_db.vis.add(db.ids[i])
                vis_id.add(db.ids[i])

    if len(ret_db.ids) == 0:
        logging.warning("No document found.\n\n")
        print("No document found.\n\n")
    return ret_db


def extract_db_by_ids(db: dbs, id_list: Union[List[str], Set[str]]) -> dbs:
    """extract db from dbs based on ids

    Args:
        db (dbs): dbs object
        id_list (Union[List[str], Set[str]]): list of ids

    Returns:
        dbs: dbs object
    """
    ret_db = dbs()
    if isinstance(id_list, list):
        id_list = set(id_list)

    for idx, id in enumerate(db.ids):
        if id in id_list:
            ret_db.ids.append(db.ids[idx])
            ret_db.embeds.append(db.embeds[idx])
            ret_db.metadatas.append(db.metadatas[idx])
            ret_db.docs.append(db.docs[idx])
            ret_db.vis.add(db.ids[idx])

    if len(ret_db.ids) == 0:
        logging.warning("No document found.\n\n")
        print("No document found.\n\n")
    return ret_db


def pop_db_by_ids(db: dbs, id_list: Union[List[str], Set[str]]):
    """pop undesired data from  dbs based on ids

    Args:
        db (dbs): dbs object
        id_list (Union[List[str], Set[str]]): list of ids

    Returns:
    """
    if isinstance(id_list, list):
        id_list = set(id_list)

    # Filter the lists in place
    db.ids = [id for id in db.ids if id not in id_list]
    db.embeds = [embed for id, embed in zip(db.ids, db.embeds) if id not in id_list]
    db.metadatas = [
        metadata for id, metadata in zip(db.ids, db.metadatas) if id not in id_list
    ]
    db.docs = [doc for id, doc in zip(db.ids, db.docs) if id not in id_list]
    db.vis = {id for id in db.vis if id not in id_list}

    if len(db.ids) == 0:
        logging.warning("No document found.\n\n")
        print("No document found.\n\n")
