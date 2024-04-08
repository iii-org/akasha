from typing import Union, List
from tqdm import tqdm
import time, os, shutil, traceback, logging
import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from pathlib import Path
import akasha.helper as helper
import akasha.akasha


class dbs:

    def __init__(self, chrdb=[]):

        self.ids = []
        self.embeds = []
        self.metadatas = []
        self.docs = []
        self.vis = set()
        if isinstance(chrdb, list):
            pass

        else:
            data = chrdb.get(include=["embeddings", "metadatas", "documents"])
            if "ids" in data:
                self.ids = data["ids"]
                self.vis = set(data["ids"])
            if "embeddings" in data:
                self.embeds = data["embeddings"]
            else:
                self.embeds = [[] for _ in range(len(data["ids"]))]
            if "metadatas" in data:
                self.metadatas = data["metadatas"]
            else:
                self.metadatas = [{} for _ in range(len(data["ids"]))]
            if "documents" in data:
                self.docs = data["documents"]
            else:
                self.docs = ["" for _ in range(len(data["ids"]))]

    def merge(self, db):

        for i in range(len(db.ids)):
            if db.ids[i] not in self.vis:
                self.ids.append(db.ids[i])
                self.embeds.append(db.embeds[i])
                self.metadatas.append(db.metadatas[i])
                self.docs.append(db.docs[i])
                self.vis.add(db.ids[i])
        # self.ids.extend(db.ids)
        # self.embeds.extend(db.embeds)
        # self.metadatas.extend(db.metadatas)
        # self.docs.extend(db.docs)

    def add_chromadb(self, chrdb):
        data = chrdb.get(include=["embeddings", "metadatas", "documents"])
        if "ids" in data:
            self.ids.extend(data["ids"])

        if "embeddings" in data:
            self.embeds.extend(data["embeddings"])
        else:
            self.embeds.extend([[] for _ in range(len(data["ids"]))])
        if "metadatas" in data:
            self.metadatas.extend(data["metadatas"])
        else:
            self.metadatas.extend([{} for _ in range(len(data["ids"]))])
        if "documents" in data:
            self.docs.extend(data["documents"])
        else:
            self.docs.extend(["" for _ in range(len(data["ids"]))])

    def get_Documents(self):
        return [
            Document(page_content=self.docs[i], metadata=self.metadatas[i])
            for i in range(len(self.docs))
        ]

    def get_docs(self):
        return self.docs

    def get_ids(self):
        return self.ids

    def get_metadatas(self):
        return self.metadatas

    def get_embeds(self):
        return self.embeds


def change_text_to_doc(texts: list):
    return [
        Document(page_content=texts[i], metadata={'page': i})
        for i in range(len(texts))
    ]


def _load_file(file_path: str, extension: str):
    """get the content and metadata of a text file (.pdf, .docx, .txt, .csv, .pptx) and return a Document object

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
            docs = CSVLoader(file_path).load()
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

            logging.warning("\nLoad " + file_path + " failed, ignored.\n" +
                            trace_text + "\n\n" + err)
        except:
            logging.warning("\nLoad file" + " failed, ignored.\n" +
                            trace_text + "\n\n" + err)
        return ""


def _load_files(doc_path: str, extension: str = "pdf") -> list:
    """get the list of text files with extension type in doc_path directory

    Args:
        **doc_path (str)**: text files directory\n
        **extension (str, optional):** the extension type. Defaults to "pdf".\n

    Returns:
        list: list of filenames
    """
    dir = Path(doc_path)
    pdf_files = dir.glob("*." + extension)
    loaders = [file.name for file in pdf_files]

    return loaders


def get_docs_from_doc(doc_path: str, chunk_size: int, ignore_check: bool):
    """get all documents from doc_path directory and split them into sentences with chunk_size

    Args:
        doc_path (str): docuemtns directory
        chunk_size (int): the chunk size of documents

    Returns:
        list: list of Documents
    """
    ignored_files = []
    # Split the documents into sentences
    documents, files = [], []
    texts = []
    db_dir = doc_path.split("/")[-2].replace(" ", "").replace(".", "")
    txt_extensions = ["pdf", "md", "docx", "txt", "csv", "pptx"]
    for extension in txt_extensions:
        files.extend(_load_files(doc_path, extension))

    progress = tqdm(total=len(files), desc="Vec Storage")

    for file in files:

        exist = False
        progress.update(1)
        if ignore_check:  ## if ignore_check is True, don't load file text and check md5 if the db is existed, to increase loading speed ##
            storage_directory, exist = check_db_name(file, db_dir, '*', '*',
                                                     chunk_size)
        if exist:
            temp_chroma = Chroma(persist_directory=storage_directory)
            if temp_chroma is not None:
                temp_docs = temp_chroma.get(include=["documents", "metadatas"])
                texts.extend([
                    Document(page_content=temp_docs["documents"][i],
                             metadata=temp_docs["metadatas"][i])
                    for i in range(len(temp_docs["documents"]))
                ])
                del temp_chroma, temp_docs
            continue

        ## if ignore_check is false or no db found, load file text and split into sentences ##
        temp_docs = _load_file(doc_path + file, file.split(".")[-1])
        if temp_docs == "" or len(temp_docs) == 0:
            ignored_files.append(file)
            continue
        documents.extend(temp_docs)

    progress.close()
    try:
        if len(documents) == 0 and len(texts) == 0:
            raise Exception("No documents found.\n\n")
        if len(documents) != 0:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n", " ", ",", ".", "。", "!"],
                chunk_size=chunk_size,
                chunk_overlap=100,
            )
            docs = text_splitter.split_documents(documents)
            texts.extend(docs)
        if len(texts) == 0:
            raise Exception("No texts found.\n\n")
    except Exception as e:
        logging.warning("\nLoad " + doc_path + " failed, ignored.\n" + str(e))
        return None, ignored_files

    return texts, ignored_files


def get_chromadb_from_file(documents: list,
                           storage_directory: str,
                           chunk_size: int,
                           embeddings: vars,
                           file_name: str,
                           sleep_time: int = 60,
                           add_pic: bool = False,
                           embed_type: str = ""):
    """load the existing chromadb of documents from storage_directory and return it, if not exist, create it.

    Args:
        documents (list): list of Documents
        storage_directory (str): the path of chromadb directory
        chunk_size (int): the chunk size of documents
        embeddings (vars): the embeddings used in transfer documents into vector storage
        file_name (str): the path and name of the file
        sleep_time (int, optional): waiting time if exceed api calls. Defaults to 60.
        add_pic (bool, optional): add pic summary in doc file into db or not. Defaults to False.

    Returns:
        (chromadb object, bool): return the chromadb object and add_pic flag
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ",", ".", "。", "!"],
        chunk_size=chunk_size,
        chunk_overlap=100,
    )
    k = 0
    cum_ids = 0
    interval = 3
    if Path(storage_directory).exists():
        docsearch = Chroma(persist_directory=storage_directory,
                           embedding_function=embeddings)
        db = dbs(docsearch)
        docsearch._client._system.stop()
        docsearch = None
        del docsearch

    else:
        docsearch = Chroma(persist_directory=storage_directory,
                           embedding_function=embeddings)
        if embed_type == "openai":
            open_model = helper.handle_model("openai:gpt-3.5-turbo", False,
                                             0.0)
        else:
            open_model = None

        while k < len(documents):
            cur_doc = documents[k:k + interval]
            texts = text_splitter.split_documents(cur_doc)
            formatted_date = datetime.datetime.now().strftime(
                "%Y-%m-%d-%H_%M_%S_%f")

            ### if page_content is too long, use llm to summarize ###
            page_contents = [text.page_content for text in texts]
            if open_model != None:
                for ix, text in enumerate(texts):
                    if open_model.get_num_tokens(text.page_content) > 2000:
                        try:
                            page_contents[ix] = helper.call_model(open_model,"use traditional chinese to list details of below article:\n\n"\
                                + text.page_content + "\n\n")
                            logging.warning(
                                "\ncontent too long, using llm to summarize for embedding...\n\n"
                            )
                        except:
                            pass

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
                    , ids=[formatted_date + "_" + str(cum_ids + i) for i in range(len(texts))]
            )
            k += interval
            cum_ids += len(texts)

        ### add pic summary to db ###
        # if add_pic and file_name.split(".")[-1] == "pdf":
        #     docsearch, add_pic = add_pic_summary_to_db(docsearch, file_name, chunk_size)
        docsearch.persist()
        db = dbs(docsearch)
        docsearch._client._system.stop()
        docsearch = None
        del docsearch
    if len(db.get_ids()) == 0:
        logging.warning(f"\n{file_name} has empty content, ignored.\n")
        shutil.rmtree(storage_directory)
        return file_name, add_pic
    return db, add_pic


def processMultiDB(
    doc_path_list: Union[List[str], str],
    verbose: bool,
    embeddings: vars,
    embeddings_name: str,
    chunk_size: int,
    ignore_check: bool = False,
):
    """get all db and combine to list of db from doc_path_list if embeddings is embedding models, else return list of documents.

    Args:
        doc_path_list (Union[List[str], str]): list of directory includes documents ["doc_dir1","doc_dir2",...] or a single string of directory "doc_dir1"\n
        verbose (bool): print out logs or not\n
        embeddings (vars): the embeddings used in transfer documents into vector storage; if is rerank, embeddings should be a str\n
        embeddings_name (str): the name of embeddings\n
        chunk_size (int): the chunk size of documents\n

    Returns:
        (Optional[Documents, Chromadb, None], List[str]): return list of documents if embeddings is a str, else return list of chromadb, and list of chromadb path names
    """
    ignored_files = []
    ## if doc_path_list is a str, juest call create_chromadb function ##
    if isinstance(doc_path_list, str):
        doc_path_list = [doc_path_list]
    if len(doc_path_list) == 0:
        logging.error("\nCannot get file path.\n\n")
        raise Exception("\nCannot get any file path.\n\n")

    ## if using rerank, extend all the documents into one list ##
    if isinstance(embeddings, str):
        texts = []
        for doc_path in doc_path_list:
            cur_texts, cur_ignores = create_chromadb(doc_path,
                                                     verbose,
                                                     embeddings,
                                                     embeddings_name,
                                                     chunk_size,
                                                     ignore_check=ignore_check)
            # if isinstance(temp, tuple):
            #     temp = temp[0]
            ignored_files.extend(cur_ignores)
            if cur_texts is not None:
                texts.extend(cur_texts)
        if len(texts) == 0:
            logging.error("\nCannot get any document.\n\n")
            raise Exception("\nCannot get any document.\n\n")
        return texts, ignored_files

    ## if not using rerank, create chromadb for each doc_path and merge them ##

    dby = dbs()  # list of dbs
    for doc_path in doc_path_list:

        db2, db_names = create_chromadb(doc_path,
                                        verbose,
                                        embeddings,
                                        embeddings_name,
                                        chunk_size,
                                        ignore_check=ignore_check)

        ignored_files.extend(db_names)
        if db2 is not None:
            dby.merge(db2)

    ## check if dbs has any document ##

    if len(dby.get_ids()) == 0:
        logging.error("\nCannot get any document.\n\n")
        raise Exception("\nCannot get any document.\n\n")
        #return None, []

    return dby, ignored_files


def create_chromadb(doc_path: str,
                    verbose: bool,
                    embeddings: vars,
                    embeddings_name: str,
                    chunk_size: int,
                    sleep_time: int = 60,
                    ignore_check: bool = False) -> vars:
    """If the documents vector storage not exist, create chromadb based on all .pdf files in doc_path.
        It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        **doc_path (str)**: the path of directory that store all .pdf documents\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n
        **embeddings (vars)**: the embeddings used in transfer documents into vector storage, could be openai
        tensorflow or huggingface embeddings. \n
        **sleep_time (int, optional)**: sleep time to transfer documents into vector storage, this is for
            preventing rate limit exceed when request too much tokens at a time. Defaults to 60.\n

    Raises:
        FileNotFoundError: if can not found the doc_path directory, will raise error and return None.

    Returns:
        vars: return the created chroma client. If documents vector storage already exist, load the vector storage
        and return.
    """

    ### check if doc path exist ###
    if not helper.is_path_exist(doc_path):
        logging.warning("\nCannot find the directory: " + doc_path + "\n\n")
        return None, []

    ## add '/' at the end of doc_path ##
    if doc_path[-1] != "/":
        doc_path += "/"
    db_dir = doc_path.split("/")[-2].replace(" ", "").replace(".", "")
    embed_type, embed_name = helper._separate_name(embeddings_name)

    ## if embeddings is a str, use rerank, so return docuemnts, not vectors ##
    if isinstance(embeddings, str):
        return get_docs_from_doc(doc_path, chunk_size, ignore_check)

    if embed_type == "openai":
        add_pic = True
    else:
        add_pic = False
    db_path_names = []

    dby = dbs()  # list of dbs
    files = []
    txt_extensions = ["pdf", "md", "docx", "txt", "csv", "pptx"]
    for extension in txt_extensions:
        files.extend(_load_files(doc_path, extension))
    progress = tqdm(total=len(files), desc="Vec Storage")

    for file in files:
        progress.update(1)
        exist = False
        if ignore_check:  ## if ignore_check is True, don't load file text and check md5 if the db is existed, to increase loading speed ##
            storage_directory, exist = check_db_name(file, db_dir, embed_type,
                                                     embed_name, chunk_size)
        if exist:
            temp_chroma = Chroma(persist_directory=storage_directory)
            if temp_chroma is not None:
                dby.add_chromadb(temp_chroma)
                #db_path_names.append(storage_directory)
                del temp_chroma
            else:
                db_path_names.append(file)
            continue

        file_doc = _load_file(doc_path + file, file.split(".")[-1])
        if file_doc == "" or len(file_doc) == 0:
            continue
        md5_hash = helper.get_text_md5("".join(
            [fd.page_content for fd in file_doc]))

        storage_directory = (
            "chromadb/" + db_dir + "_" +
            file.split(".")[0].replace(" ", "").replace("_", "") + "_" +
            md5_hash + "_" + embed_type + "_" + embed_name.replace("/", "-") +
            "_" + str(chunk_size))

        db, add_pic = get_chromadb_from_file(file_doc, storage_directory,
                                             chunk_size, embeddings,
                                             doc_path + file, sleep_time,
                                             add_pic, embed_type)

        if isinstance(db, str):
            db_path_names.append(db)
        else:
            dby.merge(db)

    progress.close()

    if len(dby.get_ids()) == 0:
        return None, db_path_names

    info = "\n\nload files:" + str(len(files)) + "\n\n"
    if verbose:
        print(info)

    return dby, db_path_names


def get_db_from_chromadb(db_path_list: list, embedding_name: str):
    """_summary_

    Args:
        db_path_list (list): list of chromadb path names
        embedding_name (str): the name of embeddings

    Returns:
        (List[Union[Document,chromadb]], List[str]): return list of documents or list of chromadb, and list of chromadb path names
    """
    ### CHROMADB_PATH = "chromadb/"

    progress = tqdm(total=len(db_path_list), desc="Vec Storage")
    ignored_files = []
    if "rerank" in embedding_name:
        texts = []
        for doc_path in db_path_list:
            progress.update(1)
            temp = Chroma(persist_directory=doc_path).get(
                include=["documents", "metadatas"])
            db = [
                Document(page_content=temp["documents"][i],
                         metadata=temp["metadatas"][i])
                for i in range(len(temp["documents"]))
            ]
            if temp is None or ''.join([d.page_content for d in db]) == "":
                ignored_files.append(doc_path)
            else:
                texts.extend(db)

        if len(texts) == 0:
            progress.close()
            logging.error("\nCannot get any document.\n\n")
            raise Exception("\nCannot get any document.\n\n")
            #return None, []
        progress.close()
        return texts, ignored_files

    dby = dbs()
    for db_path in db_path_list:
        progress.update(1)
        doc_search = Chroma(persist_directory=db_path)
        db = dbs(doc_search)

        if db is None or ''.join(db.get_docs()) == "":
            logging.warning("Cannot get any text from " + db_path + "\n\n")
            ignored_files.append(db_path)
        else:
            dby.merge(db)

        doc_search._client._system.stop()
        doc_search = None
        del db, doc_search
    progress.close()

    if len(dby.get_ids()) == 0:
        logging.error("\nCannot get any document.\n\n")
        raise Exception("\nCannot get any document.\n\n")
        # return None, []

    return dby, ignored_files


def create_single_file_db(file_path: str,
                          embeddings_name: str,
                          chunk_size: int,
                          sleep_time: int = 60):
    try:
        if isinstance(file_path, Path):
            doc_path = str(file_path.parent).replace("\\", "/")
            file_name = file_path.name
        else:
            doc_path = "/".join(file_path.split("/")[:-1])
            file_name = file_path.split("/")[-1]
    except:
        logging.warning("file path error.\n\n")
        return False, "file path error.\n\n"

    ## add '/' at the end of doc_path ##
    if doc_path[-1] != "/":
        doc_path += "/"
    db_dir = doc_path.split("/")[-2].replace(" ", "").replace(".", "")
    add_pic = True
    embed_type, embed_name = helper._separate_name(embeddings_name)
    embeddings_obj = helper.handle_embeddings(embeddings_name, True)

    file_doc = _load_file(doc_path + file_name, file_name.split(".")[-1])
    if file_doc == "" or len(file_doc) == 0:
        logging.warning("file load failed or empty.\n\n")
        return False, "file load failed or empty.\n\n"

    md5_hash = helper.get_text_md5("".join(
        [fd.page_content for fd in file_doc]))

    storage_directory = (
        "chromadb/" + db_dir + "_" +
        file_name.split(".")[0].replace(" ", "").replace("_", "") + "_" +
        md5_hash + "_" + embed_type + "_" + embed_name.replace("/", "-") +
        "_" + str(chunk_size))

    db, add_pic = get_chromadb_from_file(file_doc, storage_directory,
                                         chunk_size, embeddings_obj,
                                         doc_path + file_name, sleep_time,
                                         add_pic, embed_type)

    if isinstance(db, str):
        del embeddings_obj
        logging.warning(f"create chromadb {db} failed.\n\n")
        return False, f"create chromadb {db} failed.\n\n"
    del embeddings_obj, db

    return True, storage_directory


def save_pdf_pic(file_path: str):
    import fitz  # PyMuPDF
    import os

    output_list, mid_names, pages = [], [], []
    # create a temp_pic directory to store images if not exist
    if not os.path.exists("temp_pic"):
        os.makedirs("temp_pic")

    mid_name = os.path.splitext(os.path.split(file_path)[1])[0]
    doc = fitz.open(file_path)

    for i in range(len(doc)):
        for img in doc.get_page_images(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            ## if pic size smaller than 10 kb, ignore it ##
            if len(pix.tobytes()) < 10000:
                continue

            output_name = "./temp_pic/" + mid_name + "_" + str(i) + ".png"
            mid_names.append(mid_name)
            pages.append(i)
            output_list.append(output_name)
            if pix.n < 5:  # this is GRAY or RGB
                pix.save(output_name, "png")
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(output_name, "png")
                pix1 = None
            pix = None

    return output_list, mid_names, pages


def add_pic_summary_to_db(db, file_path, chunk_size):
    add_pic = True

    file_pics, mid_names, pages = save_pdf_pic(file_path)
    get_pic_sum_prompt = akasha.prompts.format_pic_summary_prompt(chunk_size)
    try:
        for i, file_pic in enumerate(file_pics):
            cur_summary = akasha.akasha.openai_vision(file_pic,
                                                      get_pic_sum_prompt)
            db.add_texts(
                [cur_summary],
                metadatas=[{
                    "source": mid_names[i],
                    "page": pages[i]
                }],
                ids=[mid_names[i] + "_" + str(pages[i])],
            )
    except:
        print("openai vision failed")
        add_pic = False

    db.persist()

    ## delete temp_pic directory ##
    import shutil

    shutil.rmtree("temp_pic")

    return db, add_pic


def check_db_name(file, db_dir, embed_type, embed_name, chunk_size):

    storage_directory = ("chromadb/" + db_dir + "_" +
                         file.split(".")[0].replace(" ", "").replace("_", "") +
                         "_*_" + embed_type + "_" +
                         embed_name.replace("/", "-") + "_" + str(chunk_size))
    # Use the glob method to find directories that match the pattern
    matching_dirs = list(Path().glob(storage_directory))

    if matching_dirs:
        # Sort the directories by modification time
        matching_dirs.sort(key=os.path.getmtime)

        # Keep the last directory (the one with the latest modification time)
        latest_dir = matching_dirs[-1]
        try:
            # Remove all other directories
            for dir in matching_dirs[:-1]:
                shutil.rmtree(dir)
        except:
            pass

        return str(latest_dir), True

    else:
        return storage_directory, False
