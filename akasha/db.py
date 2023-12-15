from typing import Union, List
from tqdm import tqdm
import time
import datetime
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma, chroma
from langchain.docstore.document import Document
from pathlib import Path
import json
import akasha.helper as helper
import akasha.akasha


def _load_file(file_path: str, extension: str):
    """get the content and metadata of a text file (.pdf, .docx, .txt, .csv) and return a Document object

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

        else:
            docs = TextLoader(file_path, encoding="utf-8").load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = i

        if len(docs) == 0:
            raise Exception

        return docs
    except Exception as err:
        print("Load", file_path, "failed, ignored.\n message: \n", err)
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


def get_docs_from_doc(doc_path: str, chunk_size: int):
    """get all documents from doc_path directory and split them into sentences with chunk_size

    Args:
        doc_path (str): docuemtns directory
        chunk_size (int): the chunk size of documents

    Returns:
        list: list of Documents
    """
    # Split the documents into sentences
    documents = []
    txt_extensions = ["pdf", "md", "docx", "txt", "csv"]
    for extension in txt_extensions:
        for file in _load_files(doc_path, extension):
            documents.extend(_load_file(doc_path + file, extension))
    if len(documents) == 0:
        return None
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ",", ".", "。", "!"],
        chunk_size=chunk_size,
        chunk_overlap=40,
    )
    docs = text_splitter.split_documents(documents)
    texts = [doc for doc in docs]
    if len(texts) == 0:
        return None
    return texts


def get_chromadb_from_file(
    documents: list,
    storage_directory: str,
    chunk_size: int,
    embeddings: vars,
    file_name: str,
    sleep_time: int = 60,
    add_pic: bool = False,
):
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
        chunk_overlap=40,
    )
    k = 0
    cum_ids = 0
    interval = 3
    if Path(storage_directory).exists():
        pass

    else:
        while k < len(documents):
            cur_doc = documents[k : k + interval]
            texts = text_splitter.split_documents(cur_doc)
            formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")
            try:
                if k == 0:
                    if len(texts) != 0:
                        docsearch = Chroma.from_documents(
                            texts,
                            embeddings,
                            persist_directory=storage_directory,
                            ids=[
                                formatted_date + "_" + str(cum_ids + i)
                                for i in range(len(texts))
                            ],
                        )
                    else:
                        docsearch = Chroma(
                            persist_directory=storage_directory,
                            embedding_function=embeddings,
                        )

                else:
                    if len(texts) != 0:
                        docsearch.add_documents(
                            texts,
                            ids=[
                                formatted_date + "_" + str(cum_ids + i)
                                for i in range(len(texts))
                            ],
                        )
            except:
                time.sleep(sleep_time)
                if k == 0:
                    if len(texts) != 0:
                        docsearch = Chroma.from_documents(
                            texts,
                            embeddings,
                            persist_directory=storage_directory,
                            ids=[
                                formatted_date + "_" + str(cum_ids + i)
                                for i in range(len(texts))
                            ],
                        )
                    else:
                        docsearch = Chroma(
                            persist_directory=storage_directory,
                            embedding_function=embeddings,
                        )

                else:
                    if len(texts) != 0:
                        docsearch.add_documents(
                            texts,
                            ids=[
                                formatted_date + "_" + str(cum_ids + i)
                                for i in range(len(texts))
                            ],
                        )

            k += interval
            cum_ids += len(texts)

        ### add pic summary to db ###
        # if add_pic and file_name.split(".")[-1] == "pdf":
        #     docsearch, add_pic = add_pic_summary_to_db(docsearch, file_name, chunk_size)
        docsearch.persist()
        del docsearch

    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
    temp_docs = db.get(include=["metadatas"])["metadatas"]
    # print(db.get(include=["documents"])["documents"])

    if len(temp_docs) == 0:
        print("Can not load file:", file_name)
        return None, add_pic
    return db, add_pic


def processMultiDB(
    doc_path_list: Union[List[str], str],
    verbose: bool,
    embeddings: vars,
    embeddings_name: str,
    chunk_size: int,
    use_whole_dir: bool = False,
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
    ## if doc_path_list is a str, juest call create_chromadb function ##
    if isinstance(doc_path_list, str):
        doc_path_list = [doc_path_list]
    if len(doc_path_list) == 0:
        return None, []

    ## if using rerank, extend all the documents into one list ##
    if isinstance(embeddings, str):
        texts = []
        for doc_path in doc_path_list:
            temp = create_chromadb(
                doc_path, verbose, embeddings, embeddings_name, chunk_size
            )
            if isinstance(temp, tuple):
                temp = temp[0]
            if temp is not None:
                texts.extend(temp)
        if len(texts) == 0:
            return None, []
        return texts, []

    ## if not using rerank, create chromadb for each doc_path and merge them ##

    # dbs = Chroma(embedding_function=embeddings)
    dbs = []  # list of dbs
    db_path_names = []
    for doc_path in doc_path_list:
        if use_whole_dir:
            db2 = [create_chromadb_use_dir(doc_path, verbose, embeddings, embeddings_name, chunk_size)]
        else:
            db2, db_names = create_chromadb(doc_path, verbose, embeddings, embeddings_name, chunk_size)

        if db2 is not None:
            dbs.extend(db2)
            if not use_whole_dir:
                db_path_names.extend(db_names)
    ## check if dbs has any document ##

    if len(dbs) == 0:
        return None, []

    return dbs, db_path_names


def create_chromadb(
    doc_path: str,
    verbose: bool,
    embeddings: vars,
    embeddings_name: str,
    chunk_size: int,
    sleep_time: int = 60,
) -> vars:
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
        return None, []

    ## add '/' at the end of doc_path ##
    if doc_path[-1] != "/":
        doc_path += "/"
    db_dir = doc_path.split("/")[-2].replace(" ", "").replace(".", "")
    embed_type, embed_name = helper._separate_name(embeddings_name)

    ## if embeddings is a str, use rerank, so return docuemnts, not vectors ##
    if isinstance(embeddings, str):
        return get_docs_from_doc(doc_path, chunk_size)

    if embed_type == "openai":
        add_pic = True
    else:
        add_pic = False
    db_path_names = []

    dby = []  # list of dbs
    files = []
    txt_extensions = ["pdf", "md", "docx", "txt", "csv"]
    for extension in txt_extensions:
        files.extend(_load_files(doc_path, extension))
    progress = tqdm(total=len(files), desc="Vec Storage")

    for file in files:
        progress.update(1)
        file_doc = _load_file(doc_path + file, file.split(".")[-1])
        if file_doc == "" or len(file_doc) == 0:
            continue
        md5_hash = helper.get_text_md5("".join([fd.page_content for fd in file_doc]))

        storage_directory = (
            "chromadb/"
            + db_dir
            + "_"
            + md5_hash
            + "_"
            + embed_type
            + "_"
            + embed_name.replace("/", "-")
            + "_"
            + str(chunk_size)
        )

        db, add_pic = get_chromadb_from_file(
            file_doc,
            storage_directory,
            chunk_size,
            embeddings,
            doc_path + file,
            sleep_time,
            add_pic,
        )

        if db is not None:
            dby.append(db)
            db_path_names.append(storage_directory)
        # if db is not None:
        #     db_data = db.get(include=['documents','metadatas','embeddings'])
        #     dby._collection.add(
        #         embeddings=db_data['embeddings'],
        #         metadatas=db_data['metadatas'],
        #         documents=db_data['documents'],
        #         ids=db_data['ids']
        #     )
        #     db_path_names.append(storage_directory)
        #     del db
    progress.close()

    if len(dby) == 0:
        return None, []

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
    if "rerank" in embedding_name:
        texts = []
        for doc_path in db_path_list:
            temp = Chroma(persist_directory=doc_path).get(
                include=["documents", "metadatas"]
            )
            db = [
                Document(
                    page_content=temp["documents"][i], metadata=temp["metadatas"][i]
                )
                for i in range(len(temp["documents"]))
            ]
            if temp is not None:
                texts.extend(db)
        if len(texts) == 0:
            return None, []
        return texts, []

    dby = []
    for db_path in db_path_list:
        db = Chroma(persist_directory=db_path)
        if db is not None:
            dby.append(db)
        
    if len(dby) == 0:
        return None, []

    return dby, db_path_list




def create_chromadb_use_dir(doc_path:str, verbose:bool, embeddings:vars, embeddings_name:str,chunk_size:int,\
                    sleep_time:int = 60) -> vars:
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
        
        return None
    
    if doc_path[-1] != '/':
        doc_path += '/'
    
    
    ## if embeddings is a str, use rerank, so return docuemnts, not vectors ##
    if isinstance(embeddings, str):
        return get_docs_from_doc(doc_path, chunk_size)
    
    
    embed_type, embed_name = embeddings_name.split(':')
    storage_directory = 'chromadb/' + doc_path.split('/')[-2] + '_' + embed_type + '_' + embed_name.replace('/','-') + '_' + str(chunk_size)


    if Path(storage_directory).exists():
        info = "storage db already exist.\n"
        


    else:
        

        documents = []
        txt_extensions = ['pdf', 'md','docx','txt','csv']
        for extension in txt_extensions:
            file_names = _load_files(doc_path, extension)
            progress_ext = tqdm(total=len(file_names), desc="Load Files")
            for loader in file_names:
                progress_ext.update(1)
                temp = _load_file(doc_path + loader, extension)
                if temp != "" :
                    documents.extend(temp)
            
        progress_ext.close()
        info = "\n\nload files:" +  str(len(documents)) + "\n\n" 
        
        if len(documents) == 0 :
            return None


        
        
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap=40)
        k = 0
        cum_ids = 0
        interval = 5
        progress = tqdm(total = len(documents), desc="Vec Storage")
        while k < len(documents):
            
            
            progress.update(min(interval,len(documents)-k))
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)
            try :
                if k==0:
                    docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                    
                else:
                    docsearch.add_documents(texts)
            except:
                time.sleep( sleep_time )
                if k==0:
                    docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                    
                else:
                    docsearch.add_documents(texts)
                    
            k += interval
            cum_ids += len(texts)
            
            
        docsearch.persist()  
        progress.close()
    
    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
    if verbose:
        print(info)
    
    
    return db














def create_single_file_db(
    file_path: str, embeddings_name: str, chunk_size: int, sleep_time: int = 60
):
    try:
        if isinstance(file_path, Path):
            doc_path = str(file_path.parent).replace("\\", "/")
            file_name = file_path.name
        else:
            doc_path = "/".join(file_path.split("/")[:-1])
            file_name = file_path.split("/")[-1]
    except:
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
        return False, "file load failed or empty.\n\n"

    md5_hash = helper.get_text_md5("".join([fd.page_content for fd in file_doc]))

    storage_directory = (
        "chromadb/"
        + db_dir
        + "_"
        + md5_hash
        + "_"
        + embed_type
        + "_"
        + embed_name.replace("/", "-")
        + "_"
        + str(chunk_size)
    )

    db, add_pic = get_chromadb_from_file(
        file_doc,
        storage_directory,
        chunk_size,
        embeddings_obj,
        doc_path + file_name,
        sleep_time,
        add_pic,
    )

    if db is None:
        del embeddings_obj
        return False, "create chromadb failed.\n\n"
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
            cur_summary = akasha.akasha.openai_vision(file_pic, get_pic_sum_prompt)
            db.add_texts(
                [cur_summary],
                metadatas=[{"source": mid_names[i], "page": pages[i]}],
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
