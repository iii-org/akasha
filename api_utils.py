
from pathlib import Path
import requests
import os
import shutil
import time
import json
import hashlib


HOST = "http://127.0.0.1"
PORT = "8000"
api_url = {
    'delete_expert':  f'{HOST}:{PORT}/expert/delete',
    'test_openai': f'{HOST}:{PORT}/openai/test_openai',
    'test_azure': f'{HOST}:{PORT}/openai/test_azure'
}
DOCS_PATH = './docs'
CONFIG_PATH = './config'
EXPERT_CONFIG_PATH = './config/experts'
DATASET_CONFIG_PATH = "./config/datasets/"
DB_PATH = './chromadb/'
DEFAULT_CONFIG = {'language_model':"openai:gpt-3.5-turbo",
            'search_type': "svm",
            'top_k': 5,
            'threshold': 0.1,
            'max_token': 3000,
            'temperature':0.0,
            'use_compression':0, # 0 for False, 1 for True
            'compression_language_model':""}


def get_docs_path():
    return DOCS_PATH

def get_db_path():
    return DB_PATH

def get_default_config():
    
    return DEFAULT_CONFIG

def get_config_path():
    return CONFIG_PATH

def get_expert_config_path():
    return EXPERT_CONFIG_PATH

def get_dataset_config_path():
    return DATASET_CONFIG_PATH

def get_model_path():
    return './model'



def _separate_name(name:str):
    """ separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n 

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(':')
    if len(sep) != 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ''
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name




def generate_hash(owner:str, dataset_name:str)->str:
    """use hashlib sha256 to generate hash value of owner and dataset_name/expert_name {owner}-{dataset_name/expert_name}

    Args:
        owner (str): owner name
        dataset_name (str): dataset name or expert name

    Returns:
        str: hash value
    """
    combined_string = f"{owner}-{dataset_name}"
    sha256 = hashlib.sha256()
    sha256.update(combined_string.encode('utf-8'))
    hex_digest = sha256.hexdigest()
    return hex_digest



    



def check_dir(path:str)->bool:
    """check if directory exist, if not, create it."""
    try:
        p = Path(path)
        if not p.exists():
            p.mkdir()
    except:
        return False
    return True


def check_config(path:str)-> bool:
    """check if config directory exist, if not, create it.
    also check sub directory exist(path), if not, create it.

    Args:
        path (str): sub directory path

    Returns:
        bool: check and create success or not
    """
    try:
        
        config_p = Path(CONFIG_PATH)
    
        if not config_p.exists():
            config_p.mkdir()
            
        if path=="":
            return True
        if not Path(path).exists():
            Path(path).mkdir()
    except:
        return False    
    return True


def get_file_list_of_dataset(docs_path:str)->list:
    """get all file names in a dataset

    Args:
        docs_path (str): _description_

    Returns:
        list[str]: list of file names
    """
    if not Path(docs_path).exists():
        return []
    
    files = os.listdir(docs_path)
    return files



def get_lastupdate_of_file_in_dataset(docs_path:str, file_name:str)->float:
    """get the last update time of a file in dataset

    Args:
        docs_path (str): docs directory path
        file_name (str): file path

    Returns:
        (float): float number of last update time
    """
    file_path = os.path.join(docs_path, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update


def get_lastupdate_of_dataset(dataset_name:str, owner:str)->str:
    """get the last update time of each file in dataset, and find out the lastest one

    Args:
        dataset_name (str): dataset name
        owner (str): owner name

    Raises:
        Exception: cano find DOCS_PATH directory and can not create it.
        Exception: can not create owner directory in DOCS_PATH

    Returns:
        (str): return "" if no file in dataset, else return the last update time of the latest file in str format
    """
    last_updates = []
    try:
        
        if not check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
            
        owner_path = Path(DOCS_PATH) / owner
            
        if not check_dir(owner_path):

            raise Exception(f'can not create {owner} directory in {DOCS_PATH}')
        
        docs_path = os.path.join(owner_path.__str__(), dataset_name)
        dataset_files = get_file_list_of_dataset(docs_path)
        for file in dataset_files:
            last_updates.append(get_lastupdate_of_file_in_dataset(docs_path, file))
            
        if len(last_updates) == 0:
            raise Exception(f'Dataset={dataset_name} has not file.')
        
    except Exception as e:   
       
        return ""
    
    return time.ctime(max(last_updates))





def update_dataset_name_from_chromadb(old_dataset_name:str, dataset_name:str, md5_list:list):
    """change dataset name in chromadb file, in chromadn directory, change directory name of {old_dataset_name}_md5 to {dataset_name}_md5

    Args:
        old_dataset_name (str): _description_
        dataset_name (str): _description_
        md5_list (list):
    """
    
    for md5 in md5_list:
        p = Path(DB_PATH)
        tag = old_dataset_name + '_' + md5
        for file in p.glob("*"):
            if tag in file.name:
                new_name = dataset_name + '_' + md5 + '_' + '_'.join(file.name.split('_')[2:])
                file.rename(Path(DB_PATH) / new_name)
    return




def check_and_delete_files_from_expert(dataset_name:str, owner:str, delete_files:list, old_dataset_name:str)->bool:
    """check every expert if they use this dataset, if yes, delete files that are in delete_files from expert's dataset list.
    change dataset name to new dataset name if dataset_name != old_dataset_name.
        if the expert has no dataset after deletion, delete the expert file.

    Args:
        dataset_name (str): dataset name
        owner (str): owner name
        delete_files (list): list of file names that delete from dataset

    Returns:
        bool: delete any file from expert or not
    """
    
    
    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return False
    flag = False
    delete_set = set(delete_files)
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as ff:
            expert = json.load(ff)
        for dataset in expert['datasets']:
            if dataset['name'] == old_dataset_name and dataset['owner'] == owner:
                dataset['name'] = dataset_name
                for file in dataset['files']:
                    if file in delete_set:
                        dataset['files'].remove(file)
                        flag = True
                if len(dataset['files']) == 0:
                    ## delete dataset if no file in dataset
                    expert['datasets'].remove(dataset)
                break
        if len(expert['datasets']) == 0:
            ## delete file if no dataset in expert
            res = requests.post(api_url['delete_expert'], json = {'owner':expert['owner'], 'expert_name':expert['name']}).json()
        else:        
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(expert, f, ensure_ascii=False, indent=4)

    if flag:
        return True
                    
    return False
    
    
    
    
    
def check_and_delete_dataset(dataset_name:str, owner:str)->bool:
    """check every expert if they use this dataset, if yes, delete this dataset from expert's dataset list.
        if the expert has no dataset after deletion, delete the expert file.

    Args:
        dataset_name (str): dataset name
        owner (str): owner of dataset
    """
    
    
    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return False
    flag = False
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as ff:
            expert = json.load(ff)
        for dataset in expert['datasets']:
            if dataset['name'] == dataset_name and dataset['owner'] == owner:
                expert['datasets'].remove(dataset)
                flag = True
                break
        if len(expert['datasets']) == 0:
            ## delete file if no dataset in expert
            res = requests.post(api_url['delete_expert'], json = {'owner':expert['owner'], 'expert_name':expert['name']}).json()
        else:
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(expert, f, ensure_ascii=False, indent=4)
                
    if flag:
        return True
                    
    return False





def check_and_delete_chromadb(chunk_size:int, embedding_model:str, filename:str, dataset_name:str, owner:str, id:str)->str:
    """check if any of other expert use the same file with same chunk size and embedding model, if not, delete the chromadb file

    Args:
        chunk_size (int): _description_
        embedding_model (str): _description_
        filename (str): _description_
        dataset_name (str): _description_
        owner (str): _description_
        id (str): _description_

    Returns:
        _type_: _description_
    """

    embed_type, embed_name = _separate_name(embedding_model)
    
    
    
    
    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return ""
    
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as ff:
            expert = json.load(ff)
            
        if expert['chunk_size'] == chunk_size and expert['embedding_model'] == embedding_model:
            for dataset in expert['datasets']:
                if dataset['name'] == dataset_name and dataset['owner'] == owner:
                    if filename in dataset['files']:     
                        return ""
    
    ## not find, delete
    
    # get MD5 of file
    md5 = id
    
    target_path = Path(DATASET_CONFIG_PATH) / (id+'.json')
    if target_path.exists():
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for file in data['files']:
            if file['filename'] == filename:
                md5 = file['MD5']
                break
    
    db_storage_path = DB_PATH + dataset_name + '_' + md5 + '_' + embed_type + '_' + embed_name.replace('/','-') + '_' + str(chunk_size)
   
    if Path(db_storage_path).exists():
        return db_storage_path
    
    return ""



def delete_datasets_from_expert(ori_datasets:list, delete_datasets:list):
    """from ori_datasets remove files in delete_datasets

    Args:
        ori_datasets (list): original datasets in expert
        delete_datasets (list): datasets(include name, owner, file) that wants to delete from expert

    Returns:
        list: list of datasets after deletion
    """
    delete_hash = {(dataset['owner'], dataset['name']):dataset['files'] for dataset in delete_datasets}
    
    
    for dataset in ori_datasets:
        if (dataset['owner'], dataset['name']) in delete_hash:
            for file in delete_hash[(dataset['owner'], dataset['name'])]:
                if file in dataset['files']:
                    dataset['files'].remove(file)
                    
    
    return ori_datasets


def add_datasets_to_expert(ori_datasets:list, add_datasets:list)->list:
    """merge ori_datasets and add_datasets,  if dataset already exist, append files to it, else create new dataset in ori_datasets

    Args:
        ori_datasets (list): original datasets in expert
        add_datasets (list): datasets(include name, owner, files) that wants to add to expert

    Returns:
        list[dict]: merged datasets
    """
    append_hash = {(dataset['owner'], dataset['name']):dataset['files'] for dataset in ori_datasets}
    
    
    for dataset in add_datasets:
        if (dataset['owner'], dataset['name']) in append_hash:
            for file in dataset['files']:
                if file not in append_hash[(dataset['owner'], dataset['name'])]:
                    append_hash[(dataset['owner'], dataset['name'])].append(file)
        else:
            append_hash[(dataset['owner'], dataset['name'])] = dataset['files']
            
    ori_datasets = []
    
    ## rebuild ori_datasets
    for key in append_hash:
        ori_datasets.append({
            'owner':key[0],
            'name':key[1],
            'files':append_hash[key]
        })
                    
    return ori_datasets



def choose_openai_key(config_file_path:str, openai_key:str="", azure_key:str="", azure_base:str="")->dict:
    """test the openai key, azure openai key, or keys in openai.json file and choose it if valid

    Args:
        openai_key (str, optional): openai key. Defaults to "".
        azure_key (str, optional): azure openai key. Defaults to "".
        azure_base (str, optional): azure base url. Defaults to "".
    """
    openai_key = openai_key.replace(' ','')
    azure_key = azure_key.replace(' ','')
    azure_base = azure_base.replace(' ','')
    
        
    if azure_key != "" and azure_base != "":
        #res = requests.get(api_url['test_azure'], json = {'azure_key':azure_key, 'azure_base':azure_base}).json()
        #if res['status'] == 'success':
        return {'azure_key':azure_key, 'azure_base':azure_base}
    
    if openai_key != "":
        #res = requests.get(api_url['test_openai'], json = {'openai_key':openai_key}).json()
        #if res['status'] == 'success':
        return {'openai_key':openai_key}
    
    
    if Path(CONFIG_PATH).exists():
        
        file_path = Path(config_file_path)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            
            if 'azure_key' in data and 'azure_base' in data:
                #res = requests.get(api_url['test_azure'], json = {'azure_key':data['azure_key'], 'azure_base':data['azure_base']}).json()
                #if res['status'] == 'success':
                return {'azure_key':data['azure_key'], 'azure_base':data['azure_base']}
            
            if 'openai_key' in data:
                #res = requests.get(api_url['test_openai'], json = {'openai_key':data['openai_key']}).json()
                #if res['status'] == 'success':
                return {'openai_key':data['openai_key']}
                
            
                
    
    return {}


def load_openai(config:dict)->bool:
    """delete old environment variable and load new one.

    Args:
        config (dict): dictionary may contain openai_key, azure_key, azure_base.

    Returns:
        bool: load success or not
    """
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "OPENAI_API_TYPE" in os.environ:
        del os.environ["OPENAI_API_TYPE"]
    if "OPENAI_API_VERSION" in os.environ:
        del os.environ["OPENAI_API_VERSION"]

    
    if 'openai_key' in config and config['openai_key'] != "":
        
        os.environ["OPENAI_API_KEY"] = config['openai_key']
        
        return True
    
    if 'azure_key' in config and 'azure_base' in config and config['azure_key'] != "" and config['azure_base'] != "":
        os.environ["OPENAI_API_KEY"] = config['azure_key']
        os.environ["OPENAI_API_BASE"] = config['azure_base']
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        
        return True
    
    
    return False