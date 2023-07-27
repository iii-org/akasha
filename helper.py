import time
import datetime
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.chat_models import ChatOpenAI

def _check_dir_exists(doc_path:str)->bool:
    """create 'chromadb' directory if not exist, and check if the doc db storage exist
    or not. If exist, exit create_chromadb function. 

    Args:
        doc_path (str): the path of documents directory, also used to check if it's in chromabd storage. 

    Returns:
        bool: return True of False if the doc db storage exist
    """

    chroma_path = Path('chromadb')
    if not chroma_path.exists():
        chroma_path.mkdir()
    
    

    suffix_path = doc_path.split('/')[-2]
    chroma_docs_path = Path('chromadb/'+ suffix_path)

    if chroma_docs_path.exists():
        return True



def create_chromadb(doc_path:str, logs:list, verbose:bool, embeddings:vars, sleep_time:int = 60) -> vars:
    """If the documents vector storage not exist, create chromadb based on all .pdf files in doc_path.
        It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        doc_path (str): the path of directory that store all .pdf documents
        logs (list): list that store logs
        verbose (bool): print logs or not
        embeddings (vars): the embeddings used in transfer documents into vector storage, could be openai 
        tensorflow or huggingface embeddings. 
        sleep_time (int, optional): sleep time to transfer documents into vector storage, this is for 
            preventing rate limit exceed when request too much tokens at a time. Defaults to 60.

    Raises:
        FileNotFoundError: if can not found the doc_path directory, will raise error and return None.

    Returns:
        vars: return the created chroma client. If documents vector storage already exist, load the vector storage
        and return.
    """
    

    ### check if doc path exist ###
    try:
        des_path = Path(doc_path)
        if not des_path.exists:
            raise FileNotFoundError
    except FileNotFoundError as err:
        if verbose:
            print(err)
        logs.append(err)
        return None
    ### ###


    
    if doc_path[-1] != '/':
        doc_path += '/'
    storage_directory = 'chromadb/' + doc_path.split('/')[-2]


    if _check_dir_exists(doc_path):
        info = "storage db already exist.\n"
        


    else:
        
        dir = Path(doc_path)
        pdf_files = dir.glob("*.pdf")
        loaders = [file.name for file in pdf_files]

        
        documents = []
        cou = 0

        for loader in loaders:
        
            try:
                documents.extend(PyPDFLoader(doc_path+loader).load())
                
            except:
                cou+=1
        
        info = "\n\nload files:\n\n" + doc_path+"total files:" + str(len(loaders)) +\
            " .\nfailed to read:" + str(cou) + " files\n"
        



        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
        
        k = 0
        cum_ids = 0
        interval = 60
        while k < len(documents):
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)
            if k==0:
                docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                
            else:
                docsearch.add_documents(texts)
            
            k += interval
            cum_ids += len(texts)
            time.sleep( sleep_time )
        docsearch.persist()  
    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)

    if verbose:
        print(info)
    logs.append(info)
    
    return db




def handle_embeddings(embedding_name:str, logs:list, verbose:bool)->vars :
    embedding_name = embedding_name.lower()
    openAI_list = ["text-search-ada-doc-001", "text-search-ada-query-001", "text-search-babbage-doc-001"\
                   "text-search-babbage-query-001", "text-search-curie-doc-001", \
                    "text-search-curie-query-001", "text-search-davinci-doc-001", \
                    "text-search-davinci-query-001"]

    if embedding_name in ["text-embedding-ada-002" , "openai" , "openaiembeddings"]:
        embeddings = OpenAIEmbeddings()
        info = "selected openai embeddings.\n"

    elif embedding_name in openAI_list:
        embeddings = OpenAIEmbeddings(model=embedding_name)
        info = "selected openai embeddings.\n"
    
    elif embedding_name in ["huggingface" , "huggingfaceembeddings"]:
        embeddings = HuggingFaceEmbeddings()
        info = "selected hugging face embeddings.\n"

    elif embedding_name in ["sentencetransformer" , "transformer", "transformers", "sentencetransformerembeddings"]:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        info = "selected sentence transformer embeddings.\n"

    elif embedding_name in ["tf", "tensorflow", "tensorflowhub", "tensorflowhubembeddings", "tensorflowembeddings"]:
        embeddings = TensorflowHubEmbeddings()
        info = "selected tensorflow embeddings.\n"

    else:
        embeddings = OpenAIEmbeddings()
        info = "can not find the embedding, use openai as default.\n"
    
    if verbose:
        print(info)
    logs.append(info)
    return embeddings




def handle_model(model_name:str, logs:list, verbose:bool)->vars:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
        model_name (str): open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"
        logs (list): list that store logs
        verbose (bool): print logs or not

    Returns:
        vars: model client
    """
    model_name = model_name.lower()
    openai_list = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k","text-davinci-003", "text-davinci-002"]
    if model_name in openai_list:
        model = ChatOpenAI(model=model_name, temperature = 0)
        info = "selected openai model.\n"
    else:
        model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)
        info = "can not find the model, use openai as default.\n"

    if verbose:
        print(info)
    logs.append(info)

    return model

def save_logs(logs:list)->None:
    """save running logs into logs/logs_{date}.txt

    Args:
        logs (list): list that store logs
    """
    logs = '\n'.join(logs)

    cur_date = datetime.datetime.now().strftime("%Y-%m-%d")

    logs_path = Path('logs')
    if not logs_path.exists():
        logs_path.mkdir()


    file_name = f"log_{cur_date}.txt"
    file_path = Path("logs/"+file_name)

    if file_path.exists():
        with file_path.open("a", encoding='utf-8') as file:
            file.write(logs + "\n\n")
    else:
        with file_path.open("w",  encoding='utf-8') as file:
            file.write(logs + "\n\n")
    
    return
