
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
    'delete_dataset': 'http://127.0.0.1:8002/dataset/delete'
}
DOCS_PATH = './docs'
CONFIG_PATH = './config'
EXPERT_CONFIG_PATH = './config/expert'
DATASET_CONFIG_PATH = "./config/dataset/"

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
        save_path = Path(DOCS_PATH) / dataset_name
        
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
        save_path = Path(DOCS_PATH) / dataset_name
        upload_files_list = []
        delete_files_list = []
        ## rename dataset name to new_dataset_name if there're any changes
        if new_dataset_name != dataset_name:
            if (Path(DOCS_PATH) / new_dataset_name).exists():
                raise Exception(f'New Dataset Name={new_dataset_name} is already exist')
            
            ## check if save_path is already exist
            if not save_path.exists():
                raise Exception(f'Old Dataset={dataset_name} is not exist')
            else:
                save_path.rename(Path(DOCS_PATH) / new_dataset_name)
                save_path = Path(DOCS_PATH) / new_dataset_name
        
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
        save_path = Path(DOCS_PATH) / dataset_name
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


def check_config(path:str):
    try:
        config_p = Path(CONFIG_PATH)
    
        if not config_p.exists():
            config_p.mkdir()
            
        
        if not Path(path).exists():
            Path(path).mkdir()
    except:
        return False    
    return True


def get_file_list_of_dataset(docs_path:str):
    
    if not Path(docs_path).exists():
        return []
    
    files = os.listdir(docs_path)
    return files



def get_lastupdate_of_file_in_dataset(docs_path, file_name):
    file_path = os.path.join(docs_path, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update


def get_lastupdate_of_dataset(dataset_name):
    last_updates = []
    docs_path = os.path.join(DOCS_PATH, dataset_name)
    dataset_files = get_file_list_of_dataset(docs_path)
    for file in dataset_files:
        last_updates.append(get_lastupdate_of_file_in_dataset(docs_path, file))
        
    if len(last_updates) == 0:
        return 0
    
    return time.ctime(max(last_updates))




def check_and_delete_chromadb(chunk_size:int, embedding_model:str, filename:str, dataset_name:str, owner:str, id:str):
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
    print(db_storage_path)
    if Path(db_storage_path).exists():
        shutil.rmtree(Path(db_storage_path))

    return False




def delete_datasets_from_expert(ori_datasets:list, delete_datasets:list):
    """from ori_datasets remove files in delete_datasets

    Args:
        ori_datasets (dict): _description_
        delete_datasets (dict): _description_

    Returns:
        _type_: _description_
    """
    delete_hash = {(dataset['owner'], dataset['name']):dataset['files'] for dataset in delete_datasets}
    
    
    for dataset in ori_datasets:
        if (dataset['owner'], dataset['name']) in delete_hash:
            for file in delete_hash[(dataset['owner'], dataset['name'])]:
                if file in dataset['files']:
                    dataset['files'].remove(file)
                    
    
    return ori_datasets


def add_datasets_to_expert(ori_datasets:list, add_datasets:list):
    
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