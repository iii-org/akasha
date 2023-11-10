
from pathlib import Path
import streamlit as st
import requests
import os
import shutil
import time
import json

api_url = {
    'regular_consult':'http://127.0.0.1:8002/regular_consult',
    'deep_consult': 'http://127.0.0.1:8002/deep_consult',
    'create_dataset': 'http://127.0.0.1:8002/dataset/create',
    'update_dataset': 'http://127.0.0.1:8002/dataset/update',
    'delete_dataset': 'http://127.0.0.1:8002/dataset/delete',
    'delete_expert':  'http://127.0.0.1:8002/expert/delete',
    'test_openai': 'http://127.0.0.1:8002/openai/test_openai',
    'test_azure': 'http://127.0.0.1:8002/openai/test_azure'
}
DOCS_PATH = './docs'
CONFIG_PATH = './config'
EXPERT_CONFIG_PATH = './config/expert'
DATASET_CONFIG_PATH = "./config/dataset"

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







def create_dataset(dataset_name, dataset_description, uploaded_files, owner:str):
    # validate inputs
    suc_count = 0
    
    try:
        if not check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not check_dir(owner_path):
            raise Exception(f'can not create {owner} directory in {DOCS_PATH}')
        
        save_path = owner_path / dataset_name
        
        ## check if save_path is already exist
        if save_path.exists():
            raise Exception(f'Dataset={dataset_name} is already exist')
        else:
            save_path.mkdir()
        ## check file size/extension is non-empty/extremely large
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            if len(bytes_data) == 0:
                st.warning(f'File={uploaded_file.name} is empty')
                continue
            if len(bytes_data) > 100000000:
                st.warning(f'File={uploaded_file.name} is too large')
                continue
            
            with open(save_path.joinpath(uploaded_file.name), "wb") as f:
                f.write(bytes_data)
            suc_count += 1
           #  st.write("uploaded file:", uploaded_file.name)
        if suc_count == 0:
            raise Exception('No file is uploaded successfully')
        # save data file(s) to local path={new_dataset_name}
       
        data = {
            'dataset_name': dataset_name,
            'dataset_description': dataset_description,
            'owner': owner
        }
        response = requests.post(api_url['create_dataset'],json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        st.success(f'Dataset\'{dataset_name}\' has been created successfully')
    except Exception as e:
    
        st.error('Dataset creation failed, '+ e.__str__())
        return
    
    
    





def edit_dataset(dataset_name, new_dataset_name, new_description, uploaded_files, delete_file_set, owner:str):
    # validate inputs
        
    # save uploaded files to local path={new_dataset_name}
    # delete files in delete_file_set from local path={new_dataset_name}
    
    # update vector db
    ## remove vector db with name is in [delete_file_set]
    ## add vector db with name is in [uploaded_file_list]
    ### NOT_FINISHED ###

    try:
        if not check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not check_dir(owner_path):
            raise Exception(f'can not create {owner} directory in {DOCS_PATH}')
        
        save_path = owner_path / dataset_name
        upload_files_list = []
        delete_files_list = []
        ## rename dataset name to new_dataset_name if there're any changes
        if new_dataset_name != dataset_name:
            if (owner_path / new_dataset_name).exists():
                raise Exception(f'New Dataset Name={new_dataset_name} is already exist')
            
            ## check if save_path is already exist
            if not save_path.exists():
                raise Exception(f'Old Dataset={dataset_name} is not exist')
            else:
                save_path.rename(owner_path / new_dataset_name)
                save_path = owner_path / new_dataset_name
        
        ## delete files in delete_file_set from local path={new_dataset_name}
        for file in delete_file_set:
            if (save_path / file).exists():
                (save_path / file).unlink()
                delete_files_list.append(file)

        ## check file size/extension is non-empty/extremely large
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            if len(bytes_data) == 0:
                st.warning(f'File={uploaded_file.name} is empty')
                continue
            if len(bytes_data) > 100000000:
                st.warning(f'File={uploaded_file.name} is too large')
                continue
            
            with open(save_path.joinpath(uploaded_file.name), "wb") as f:
                f.write(bytes_data)
        
            upload_files_list.append(uploaded_file.name)
        
        data = {
            'dataset_name': dataset_name,
            'new_dataset_name': new_dataset_name,
            'new_dataset_description': new_description,
            'owner': owner,
            'upload_files': upload_files_list,
            'delete_files': delete_files_list
        }
        response = requests.post(api_url['update_dataset'], json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        
        st.success(f'Dataset\'{dataset_name}\' has been updated successfully')
        
        
    except Exception as e:
    
        st.error('Dataset edition failed, '+ e.__str__())
        return
    
    return 

def delete_dataset(dataset_name, owner:str):
    # validate inputs
    try:
        if not check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not check_dir(owner_path):
            raise Exception(f'can not create {owner} directory in {DOCS_PATH}')
        
        
        save_path = owner_path / dataset_name
        if not save_path.exists():
            raise Exception(f'Dataset={dataset_name} is not exist')
        
        save_path.rmdir()
        data = {
            'dataset_name': dataset_name,
            'owner': owner
        }
        response = requests.post(api_url['delete_dataset'],json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        st.success(f'Dataset\'{dataset_name}\' has been deleted successfully')
    except Exception as e:
    
        st.error('Dataset deletion failed, '+ e.__str__())
        return
    
    return

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
        path (str): _description_

    Returns:
        _type_: _description_
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


def get_file_list_of_dataset(docs_path:str):
    """get all file names in a dataset

    Args:
        docs_path (str): _description_

    Returns:
        _type_: _description_
    """
    if not Path(docs_path).exists():
        return []
    
    files = os.listdir(docs_path)
    return files



def get_lastupdate_of_file_in_dataset(docs_path, file_name)->float:
    """get the last update time of a file in dataset

    Args:
        docs_path (_type_): docs directory path
        file_name (_type_): file path

    Returns:
        _type_: _description_
    """
    file_path = os.path.join(docs_path, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update


def get_lastupdate_of_dataset(dataset_name:str, owner:str):
    """get the last update time of each file in dataset, and find out the lastest one

    Args:
        dataset_name (str): dataset name
        owner (str): owner name

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """
    last_updates = []
    
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
        return 0
    
    return time.ctime(max(last_updates))


def check_and_delete_dataset(dataset_name:str, owner:str)->bool:
    """check every expert if they use this dataset, if yes, delete this dataset from expert's dataset list.
        if the expert has no dataset after deletion, delete the expert file.

    Args:
        dataset_name (str): _description_
        owner (str): _description_
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
                if len(expert['datasets']) == 0:
                    ## delete file if no dataset in expert
                    res = requests.post(api_url['delete_expert'], json = {'owner':expert['owner'], 'expert_name':expert['name']}).json()
                else:
                    with open(file, 'w', encoding='utf-8') as f:
                        json.dump(expert, f, ensure_ascii=False, indent=4)
                flag = True
                
                    
                break
    if flag:
        return True
                    
    return False





def check_and_delete_chromadb(chunk_size:int, embedding_model:str, filename:str, dataset_name:str, owner:str, id:str)->bool:
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
        return False
    
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as ff:
            expert = json.load(ff)
            
        if expert['chunk_size'] == chunk_size and expert['embedding_model'] == embedding_model:
            for dataset in expert['datasets']:
                if dataset['name'] == dataset_name and dataset['owner'] == owner:
                    if filename in dataset['files']:     
                        return True
    
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
    
    db_storage_path = './chromadb/' + dataset_name + '_' + md5 + '_' + embed_type + '_' + embed_name.replace('/','-') + '_' + str(chunk_size)
    if Path(db_storage_path).exists():
        shutil.rmtree(Path(db_storage_path))

    return False




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



def choose_openai_key(openai_key:str="", azure_key:str="", azure_base:str="")->dict:
    """test the openai key, azure openai key, or keys in openai.json file and choose it if valid

    Args:
        openai_key (str, optional): openai key. Defaults to "".
        azure_key (str, optional): azure openai key. Defaults to "".
        azure_base (str, optional): azure base url. Defaults to "".
    """
    openai_key = openai_key.replace(' ','')
    azure_key = azure_key.replace(' ','')
    azure_base = azure_base.replace(' ','')
    if openai_key != "":
        res = requests.get(api_url['test_openai'], json = {'openai_key':openai_key}).json()
        if res['status'] == 'success':
            return {'openai_key':openai_key}
        
    if azure_key != "" and azure_base != "":
        res = requests.get(api_url['test_azure'], json = {'azure_key':azure_key, 'azure_base':azure_base}).json()
        if res['status'] == 'success':
            return {'azure_key':azure_key, 'azure_base':azure_base}
    
    if Path(CONFIG_PATH).exists():
        
        file_path = Path(CONFIG_PATH) / 'openai.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'openai_key' in data:
                res = requests.get(api_url['test_openai'], json = {'openai_key':data['openai_key']}).json()
                if res['status'] == 'success':
                    return {'openai_key':data['openai_key']}
                
            if 'azure_key' in data and 'azure_base' in data:
                res = requests.get(api_url['test_azure'], json = {'azure_key':data['azure_key'], 'azure_base':data['azure_base']}).json()
                if res['status'] == 'success':
                    return {'azure_key':data['azure_key'], 'azure_base':data['azure_base']}
                
    
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