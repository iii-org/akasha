from langchain.schema import Document
from typing import Union, List
from pathlib import Path
from akasha.utils.db.db_structure import is_url
from akasha.utils.db.file_loader import load_file, load_directory, load_url


def _get_file_dir_docs(s: Union[str, Path]) -> List[Document]:
    try:
        if isinstance(s, str):
            if len(s) < 255:
                s = Path(s)
            else:
                return [Document(page_content=s)]
    except Exception as e:
        print(f"Error loading {s}: {e}")
        return [Document(page_content=s)]

    try:
        if s.is_file():
            return load_file(s, s.__str__().split(".")[-1])
        elif s.is_dir():
            return load_directory(s)
        else:
            return [Document(page_content=s.__str__())]
    except Exception as e:
        print(f"Error loading {s}: {e}")
        return [Document(page_content=s.__str__())]


def load_docs_from_info(info: Union[str, list, Path, Document]) -> List[Document]:
    """_summary_

    Args:
        info (Union[str, list, Path, Document]): _description_

    Returns:
        List[Document]: _description_
    """
    docs = []
    if not isinstance(info, list):
        info = [info]

    for si in info:
        if isinstance(si, str):
            if si == "":
                continue
            if is_url(si):
                docs.append(load_url(si))
            else:
                docs.extend(_get_file_dir_docs(si))
        elif isinstance(si, Path):
            docs.extend(_get_file_dir_docs(si))
        elif isinstance(si, Document):
            docs.append(si)

    return docs
