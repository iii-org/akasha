from typing import Union, List
from tqdm import tqdm
import time
import datetime
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, chroma
from langchain.docstore.document import Document
from pathlib import Path
import json
import akasha.helper as helper


def _load_file(file_path, extension):
    try:
        if extension == "pdf" or  extension == "PDF":
            docs = PyPDFLoader(file_path).load()
        elif extension == "docx" or extension == "DOCX":
            docs = Docx2txtLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = i
           
        elif extension == "csv":
            docs = CSVLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = docs[i].metadata['row']
                del docs[i].metadata['row']
            
        else:
            docs = TextLoader(file_path,encoding='utf-8').load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = i
        
        if len(docs) == 0 :
            raise Exception 
        
        return docs
    except Exception as err:
        print("Load",file_path,"failed, ignored.\n message: \n",err)
        return ""


def _load_files(doc_path:str, extension:str="pdf")->list:
    """load text files of select extension into list of Documents

    Args:
        **doc_path (str)**: text files directory\n 
        **extension (str, optional):** the extension type. Defaults to "pdf".\n 

    Returns:
        list: list of Documents
    """
    dir = Path(doc_path)
    pdf_files = dir.glob("*."+extension)
    loaders = [file.name for file in pdf_files]

    return loaders
       


def _check_dir_exists(doc_path:str, embeddings_name:str, chunk_size:int)->bool:
    """create 'chromadb' directory if not exist, and check if the doc db storage exist
    or not. If exist, exit create_chromadb function. 

    Args:
        **doc_path (str)**: the path of documents directory, also used to check if it's in chromabd storage. \n 

    Returns:
        bool: return True of False if the doc db storage exist
    """

    chroma_path = Path('chromadb')
    if not chroma_path.exists():
        chroma_path.mkdir()
    
    

    suffix_path = doc_path.split('/')[-2]
    embed_type,embed_name = embeddings_name.split(':')
    chroma_docs_path = Path('chromadb/'+ suffix_path + '_' + embed_type + '_' + embed_name.replace('/','-') +'_' + str(chunk_size) ) 

    if chroma_docs_path.exists():
        return True




def get_docs_from_doc(doc_path:str, chunk_size:int):
    # Split the documents into sentences
    documents = []
    txt_extensions = ['pdf', 'md','docx','txt','csv']
    for extension in txt_extensions:
        documents.extend(_load_files(doc_path, extension))
    if len(documents) == 0 :
        return None
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap = 40)
    docs = text_splitter.split_documents(documents)
    texts = [doc for doc in docs]
    if len(texts) == 0:
        return None
    return texts



def get_chromadb_from_file(documents:list, storage_directory:str, chunk_size:int, embeddings:vars, file_name:str, sleep_time:int = 60):
    
    
    
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap=40)
    k = 0
    cum_ids = 0
    interval = 3
    if Path(storage_directory).exists():
        pass
    
    else:
        while k < len(documents):
        
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)
            formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")
            try :
                if k==0 :
                    
                    if len(texts) != 0:
                        docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory,\
                            ids = [formatted_date +'_' + str(cum_ids+i) for i in range(len(texts))]) 
                    else:
                        docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                    
                else:
                    if len(texts) != 0:
                        docsearch.add_documents(texts, ids = [formatted_date +'_' + str(cum_ids+i) for i in range(len(texts))])
            except:
                time.sleep( sleep_time )
                if k==0:
                    
                    if len(texts) != 0:
                        docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory,\
                            ids = [formatted_date +'_' + str(cum_ids+i) for i in range(len(texts))]) 
                    else:
                        docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                    
                else:
                    if len(texts) != 0:
                        docsearch.add_documents(texts, ids = [formatted_date +'_' + str(cum_ids+i) for i in range(len(texts))])
                    
            k += interval
            cum_ids += len(texts)
            
        docsearch.persist()  
        del docsearch

    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
    temp_docs = db.get(include=['metadatas'])['metadatas']
    #print(storage_directory,"SOURCE:",set([temp['source'] for temp in  temp_docs]))
    if len(temp_docs) == 0:
        print("Can not load file:", file_name)
        return None
    return db
    
    
def processMultiDB(doc_path_list: Union[List[str], str], verbose:bool, embeddings:vars, embeddings_name:str, chunk_size:int):
    
    ## if doc_path_list is a str, juest call create_chromadb function ##
    if isinstance(doc_path_list, str):
        doc_path_list = [doc_path_list]
    if len(doc_path_list) == 0:
        return None, []
    
    ## if using rerank, extend all the documents into one list ## 
    if isinstance(embeddings, str):
        texts = []
        for doc_path in doc_path_list:
            temp = create_chromadb(doc_path, verbose, embeddings, embeddings_name, chunk_size)
            if temp is not None:
                texts.extend(temp)
        if len(texts) == 0:
            return None, []
        return texts, []

    ## if not using rerank, create chromadb for each doc_path and merge them ##

    #dbs = Chroma(embedding_function=embeddings)
    dbs = [] # list of dbs
    db_path_names = []
    for doc_path in doc_path_list:
        
        db2, db_names = create_chromadb(doc_path, verbose, embeddings, embeddings_name, chunk_size)
        
        if db2 is not None:
        
            dbs.extend(db2)
            db_path_names.extend(db_names)
    ## check if dbs has any document ##

    
    if len(dbs) == 0:

        return None, []
    
    return dbs, db_path_names


def create_chromadb(doc_path:str, verbose:bool, embeddings:vars, embeddings_name:str,chunk_size:int,\
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
    
    
    ## add '/' at the end of doc_path ##
    if doc_path[-1] != '/':
        doc_path += '/'
    db_dir = doc_path.split('/')[-2].replace(' ','').replace('.','')
    embed_type, embed_name = helper._separate_name(embeddings_name)
    storage_directory = 'chromadb/' + doc_path.split('/')[-2] + '_' + embed_type + '_' + embed_name.replace('/','-') + '_' + str(chunk_size)
    db_exi = _check_dir_exists(doc_path, embeddings_name, chunk_size)
     
    
    ## if embeddings is a str, use rerank, so return docuemnts, not vectors ##
    if isinstance(embeddings, str):
        
        return get_docs_from_doc(doc_path, chunk_size)
    
    
    ## if chromadb already exist, use metadata to check which doc files has been added ##
    elif db_exi:
        info = "storage db already exist.\n"
        # db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
        # mtda = db.get(include=['metadatas'])
        
        # for source in mtda['metadatas']:
        #     vis.add(source['source'])
        # del db,mtda
      

    db_path_names = []
    formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")
    #dby = Chroma(embedding_function=embeddings, persist_directory='chromadb/' + "temp_c&r@md&" + formatted_date)
    dby = []  # list of dbs
    files = []
    txt_extensions = ['pdf', 'md','docx','txt','csv']
    for extension in txt_extensions:
        
        files.extend(_load_files(doc_path, extension))
    progress = tqdm(total = len(files), desc="Vec Storage")
    
    for file in files:
       
        
        progress.update(1)
        file_doc = _load_file(doc_path+file, file.split('.')[-1])
        if file_doc == ""  or len(file_doc) == 0:
            continue
        md5_hash = helper.get_text_md5(''.join([fd.page_content for fd in file_doc]))
        
        storage_directory = 'chromadb/' + db_dir  + '_' + md5_hash + '_' + embed_type + '_' +\
            embed_name.replace('/','-') + '_' + str(chunk_size)
        
        
        db = get_chromadb_from_file(file_doc, storage_directory, chunk_size, embeddings, file, sleep_time)
        
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
    
    info = "\n\nload files:" +  str(len(files)) + "\n\n" 
    if verbose:
        print(info)
    
    
    return dby, db_path_names



def create_dataset(doc_path: Union[List[str], str], dataset_name:str):
    
    ## check dataset.json exist or not, if not, create one ##
    dataset_path = Path('dataset.json')
    if not dataset_path.exists():
        dataset_path.touch()
        dataset_path.write_text('{}')
    else:
        datasets = dataset_path.read_text()
        if datasets == '':
            datasets = {}
        else:
            datasets = json.loads(datasets)
            
    ## if doc_path is a str, read the directory and add every file path to dataset.json ##
    if isinstance(doc_path, str):
        doc_path = [doc_path]
    #for path in doc_path:
        
    
def get_db_from_chromadb(db_path_list:list, embedding_name:str):
    
    ### CHROMADB_PATH = "chromadb/"
    if "rerank" in embedding_name:
        
        texts = []
        for doc_path in db_path_list:
            temp = Chroma(persist_directory= doc_path).get(include=['documents','metadatas'])
            db = [Document(page_content=temp['documents'][i], metadata=temp['metadatas'][i]) for i in range(len(temp['documents']))]
            if temp is not None:
                texts.extend(db)
        if len(texts) == 0:
            return None, []
        return texts, []
    
    dby = []
    for db_path in db_path_list:
        db = Chroma(persist_directory= db_path)
        if db is not None:
            dby.append(db)
    
    if len(dby) == 0:
        return None, []
    
    return dby, db_path_list



def create_single_file_db(file_path:str, embeddings_name:str, chunk_size:int, sleep_time:int = 60):
    
    try:
        doc_path = '/'.join(file_path.split('/')[:-1])
        file_name = file_path.split('/')[-1]
    except:
        return False, "file path error.\n\n"
    
    
    ## add '/' at the end of doc_path ##
    if doc_path[-1] != '/':
        doc_path += '/'
    db_dir = doc_path.split('/')[-2].replace(' ','').replace('.','')
    embed_type, embed_name = helper._separate_name(embeddings_name)
    embeddings_obj = helper.handle_embeddings(embeddings_name, True)
    
    file_doc = _load_file(doc_path+file_name, file_name.split('.')[-1])
    if file_doc == ""  or len(file_doc) == 0:
        return False, "file load failed or empty.\n\n"
    
    md5_hash = helper.get_text_md5(''.join([fd.page_content for fd in file_doc]))
    
    storage_directory = 'chromadb/' + db_dir  + '_' + md5_hash + '_' + embed_type + '_' +\
        embed_name.replace('/','-') + '_' + str(chunk_size)
    
    
    db = get_chromadb_from_file(file_doc, storage_directory, chunk_size, embeddings_obj, file_name, sleep_time)
    
    if db is None:
        del embeddings_obj
        return False, "create chromadb failed.\n\n"
    del embeddings_obj, db
    
    return True, storage_directory