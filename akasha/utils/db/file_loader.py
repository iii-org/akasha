from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from typing import Union, List
import warnings
import logging
import traceback
from akasha.helper.encoding import detect_encoding
from pathlib import Path
from .db_structure import TEXT_EXTENSIONS
from akasha.helper.crawler import get_text_from_url

warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
logging.basicConfig(level=logging.ERROR)


def load_file(file_path: str, extension: str) -> List[Document]:
    """get the content and metadata of a text file (.pdf, .docx, .md, .txt, .csv, .pptx) and return a Document object

    Args:
        file_path (str): the path of the text file\n
        extension (str): the extension of the text file\n

    Raises:
        Exception: if the length of doc is 0, raise error

    Returns:
        list: list of Docuemnts
    """
    try:
        if extension == "pdf" or extension == "PDF":
            docs = PyPDFLoader(file_path).load()
        elif extension == "docx" or extension == "DOCX":
            docs = Docx2txtLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = i

        elif extension == "csv":
            encoding = detect_encoding(file_path)
            docs = CSVLoader(file_path, encoding=encoding).load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = docs[i].metadata["row"]
                del docs[i].metadata["row"]
        elif extension == "pptx":
            docs = UnstructuredPowerPointLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = i
        else:
            docs = TextLoader(file_path, encoding="utf-8").load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = i
        if len(docs) == 0:
            raise Exception

        return docs
    except Exception as err:
        try:
            trace_text = traceback.format_exc()

            logging.warning(
                "\nLoad "
                + file_path
                + " failed, ignored.\n"
                + trace_text
                + "\n\n"
                + str(err)
            )
        except Exception:
            logging.warning("\nLoad file" + " failed, ignored.\n" + trace_text + "\n\n")
        return []


def load_directory(directory_path: Union[str, Path]) -> List[Document]:
    """get the content and metadata of text files in a directory and return a list of Document objects

    Args:
        directory_path (Union[str, Path]): the path of the directory\n

    Returns:
        list: list of Documents
    """

    ### get the chromadb directory name###
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    files, docs = [], []

    for extension in TEXT_EXTENSIONS:
        files.extend(get_load_file_list(directory_path, extension))

    for file in files:
        whole_file_path = directory_path / file
        file_doc = load_file(whole_file_path, file.split(".")[-1])
        if file_doc == "" or len(file_doc) == 0:
            logging.warning(f"file {file} load failed or empty.\n\n")
            print(f"file {file} load failed or empty.\n\n")

        docs.extend(file_doc)
    return docs


def load_url(url: str) -> Document:
    """get the content and metadata of a url and return a Document object

    Args:
        url (str): the url\n

    Returns:
        Document: Document object
    """
    url_title, url_text = get_text_from_url(url)

    return Document(page_content=url_text, metadata={"title": url_title, "url": url})


def get_load_file_list(doc_path: Union[str, Path], extension: str = "pdf") -> list:
    """get the list of text files with extension type in doc_path directory

    Args:
        **doc_path (Union[str,Path])**: text files directory\n
        **extension (str, optional):** the extension type. Defaults to "pdf".\n

    Returns:
        list: list of filenames
    """
    dir = Path(doc_path)
    pdf_files = dir.glob("*." + extension)
    loaders = [file.name for file in pdf_files]

    return loaders
