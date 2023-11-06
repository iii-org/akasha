
from pathlib import Path
import streamlit as st
import requests
import os
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




