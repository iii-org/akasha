import streamlit as st
import pandas as pd
import config
import os
from typing import List
import requests
from pathlib import Path
import streamlit_utils as stu

DATSEST_PATH = os.path.join('.', 'datasets')
CHUNKSIZE = 500


HOST = "http://127.0.0.1"
PORT = "8000"
api_urls = {
    'get_docs_path': f'{HOST}:{PORT}/get_docs_path',
    'get_dataset':f'{HOST}:{PORT}/dataset/get',
    'get_filename_list':f'{HOST}:{PORT}/dataset/get_filename',
    'show_dataset': f'{HOST}:{PORT}/dataset/show',
    'get_owner_dataset': f'{HOST}:{PORT}/dataset/get_owner',
    'share_dataset': f'{HOST}:{PORT}/dataset/share',
    'create_dataset': f'{HOST}:{PORT}/dataset/create',
    'update_dataset': f'{HOST}:{PORT}/dataset/update',
    'delete_dataset': f'{HOST}:{PORT}/dataset/delete',
    'get_expert': f'{HOST}:{PORT}/expert/get',
    'get_owner_expert': f'{HOST}:{PORT}/expert/get_owner',
    'create_expert': f'{HOST}:{PORT}/expert/create',
    'update_expert': f'{HOST}:{PORT}/expert/update',
    'delete_expert': f'{HOST}:{PORT}/expert/delete',
    'share_expert': f'{HOST}:{PORT}/expert/share',
    'show_expert': f'{HOST}:{PORT}/expert/show',
    'get_dataset_dcp': f'{HOST}:{PORT}/dataset/get_dcp',
    'test_openai': f'{HOST}:{PORT}/openai/test_openai',
    'test_azure': f'{HOST}:{PORT}/openai/test_azure',
    'save_openai': f'{HOST}:{PORT}/openai/save',
    'choose_openai': f'{HOST}:{PORT}/openai/choose'
}


def check_dataset_is_shared(dataset_name):
    return '@' in dataset_name
def check_expert_is_shared(expert_name):
    return '@' in expert_name

def getMD5(filename, path:str=os.getcwd()):
    return ''

def add_question_layer(add_button):
    
    df = pd.DataFrame(
        [
            {"Sub-Questions": ""}
        ]
    )
   
    with add_button:
        add = st.button(label="➕ Add Layer", use_container_width=True)
    if st.session_state.get('question_layers') == None:
        st.session_state.question_layers = 1
    if add:
        st.session_state.question_layers += 1
        # st.experimental_rerun()
    layers = ["" for _ in range(st.session_state.question_layers)]

    for i in range(st.session_state.question_layers):
        # col_layer_id, col_layer_data = st.columns([1, 6])
        layer_name = st.caption(f'Layer {i+1}')
        layer_content = st.data_editor(df, key=f'question-layer-{i}', num_rows='dynamic', use_container_width=True, hide_index=True)
        layers[i] = (layer_name, layer_content) #(col_layer_data, col_layer_id)
        # layers[i] = st.data_editor(df, key=f'question-layer-{i}', num_rows='dynamic', use_container_width=True)

    return layers

def ask_question(col, username, sys_prompt, prompt, expert_owner, expert_name, advanced_params:dict, auto_clean=False):
    if not prompt:
        col.error('❌ Please input question')
        return False
    with col.chat_message("user"):
        st.markdown(prompt)
    with col.chat_message("assistant"):
        # akasha: get response from expert
        success = True
        if success:
            st.write('my random answer')
            # save last consult config for expert if is the owner of expert
            if username == expert_owner:
                expert = config.expert.from_json(expert_owner, expert_name)
                expert.add_consultations(advanced_params)
                expert.save()
    if auto_clean:
        st.session_state['question'] = ''
        
def ask_question_deep(col, layers_list:List[dict], username, sys_prompt, prompt, expert_owner, expert_name, advanced_params:dict, auto_clean=False):
    if not prompt:
        col.error('❌ Please input question')
        return False
    with col.chat_message("user"):
        st.markdown(prompt)
    with col.chat_message("assistant"):
        success = True
        if success:
            # get response from expert with chain of thought
            st.write('my random answer from layers of questions')
            # save last consult config for expert if is the owner of expert
            if username == expert_owner:
                expert = config.expert.from_json(expert_owner, expert_name)
                expert.add_consultations(advanced_params)
                expert.save()
    if auto_clean:
        st.session_state['final-question'] = ''

def get_last_consult_for_expert(expert_name):
    # get last consult config for expert
    return {'language_model':'gpt2',
            'search_type':'merge',
            'top_k':3,
            'threshold':0.5,
            'max_token':50,
            'temperature':0.55,
            'use_compression':0, # 0 for False, 1 for True
            'compression_language_model':'gpt3'}

def save_last_consult_for_expert(expert_name):
    # save last consult config for expert
    return True

def list_experts(owner:str=None, name_only:bool=False, include_shared=True):
    # list all experts (of specific owner)
    if include_shared:
        response = requests.get(api_urls['get_expert'], json = {'owner':owner}).json()
        
    else:
        response = requests.get(api_urls['get_owner_expert'], json = {'owner':owner}).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])    
        return []
    
    experts = [e['dataset_name'] if e['owner'] == owner else f"{e['dataset_name']}@{e['owner']}" for e in response['response'] ]
    # if name_only:
    #     return [e['name'] if e['owner'] == owner else f"{e['name']}@{e['owner']}" for e in experts]
    return experts

def create_expert(owner:str, expert_name:str, expert_embedding:str, expert_chunksize:int,
                expert_add_files:dict):
    # # validate inputs
    # ## check chunksize is valid: not extremely large
    # if expert_chunksize > CHUNKSIZE:
    #     st.warning(f'❌ Chunksize should be less than {CHUNKSIZE}')
    #     return False
    # ## check expert name is valid: not used already
    # user_experts = list_experts(owner, name_only=True)
    # if expert_name in user_experts:
    #     st.error(f'❌ Expert={expert_name} already exists')
    #     return False
    # # save configurations
    # expert_datasets = [{'owner':owner if not check_dataset_is_shared(ds) else ds.split('@')[-1], 
    #                     'name':ds if not check_dataset_is_shared(ds) else ds.split('@')[0], 
    #                     'files':list(expert_add_files_dict.get(ds))} 
    #                    for ds in expert_datasets]
    # expert = config.expert.create(owner, expert_name, expert_embedding, expert_chunksize, expert_datasets, shared_user_accounts)
    
    # # TODO: Akasha sdk...
    
    
    try:
        if expert_embedding.split(':')[0] == "openai":
            openai_config = get_openai_config(owner)
        else:
            openai_config = {}
        datasets = []
        for k,v in expert_add_files.items():
            dataset_owner = owner if not check_dataset_is_shared(k) else k.split('@')[-1]
            dataset_name = k if not check_dataset_is_shared(k) else k.split('@')[0]
            filename_list = list(v) 
            datasets.append({'owner':dataset_owner, 'name':dataset_name, 'files':filename_list})
            
        data = {'owner':owner,
            'expert_name':expert_name,
            'embedding_model':expert_embedding,
            'chunk_size':expert_chunksize, 
            'datasets':datasets,
            'openai_config':openai_config
        }
        
        response = requests.post(api_urls['create_expert'], json = data).json()
        if response['status'] != 'success':
                raise Exception(response['response'])
            
            #st.success(f'Expert\'{expert_name}\' has been created successfully')
    
    
    except Exception as e:
    
        st.error('❌ Expert creation failed, '+ e.__str__())
        return False
    
    return True

        
def edit_expert(owner:str, expert_name:str, new_expert_name:str, 
                default_expert_embedding, new_expert_embedding, 
                default_expert_chunksize, new_expert_chunksize, 
                default_expert_datasets, new_expert_datasets,
                expert_used_dataset_files_dict,#:Dict[set],
                share_or_not:bool, shared_user_accounts:list=[]):
    # validate inputs
    ## update_expert_name is valid: not used already
    user_experts = list_experts(owner, name_only=True)
    if (new_expert_name != expert_name) and (new_expert_name in user_experts):
        st.error(f'❌ Expert={expert_name} already exists')
        return
    ## update_chunksize is valid: not extremely large
    if new_expert_chunksize > CHUNKSIZE:
        st.warning(f'❌ Chunksize should be less than {CHUNKSIZE}')
        return
    ## new_expert_datasets is valid: not empty
    if new_expert_datasets == []:
        st.error(f'❌ Expert should use at least one dataset')
        return
    ## at least one file is selected among all datasets
    for _,fileset in expert_used_dataset_files_dict.items():
        if len(fileset) > 0:
            break
        st.error(f'❌ Expert should select at least 1 file among all datasets')
        return
    ## must select at least one user to share expert when share_or_not=True
    if share_or_not:
        if len(shared_user_accounts) == 0:
            st.error(f'❌ Please select user(s) to share expert, or disable user-sharing.')
            return
    # rename expert name to new_expert_name
    expert = config.expert.from_json(owner, expert_name)
    if (expert_name != new_expert_name):
        expert.delete()
        expert.set_name(new_expert_name)
    expert.set_embedding_model(new_expert_embedding)
    expert.set_chunk_size(new_expert_chunksize)
    expert.set_uid()
    expert.clean_datasets()
    for dataset_name in new_expert_datasets:
        dataset_files = list(expert_used_dataset_files_dict.get(dataset_name, set()))
        if check_dataset_is_shared(dataset_name):
            dataset_name, dataset_owner = dataset_name.split('@')
        else:
            dataset_owner = owner
        expert.add_dataset(dataset_owner, dataset_name, dataset_files)
    
    expert.clean_shared_users()
    if share_or_not:
        expert.add_share_users(shared_user_accounts)
    
    expert.save()
    
    # akasha sdk...
    # if change embedding model or chunksize, construct new vector db 
    # update configuration of expert's vector db(s) 
    
    return True

def delete_expert(username, expert_name):
    # delete expert from all experts in config
    
    response = requests.post(api_urls['delete_expert'], json = {'owner':username, 'expert_name':expert_name}).json()
    if response['status'] != 'success':
        api_fail(response['response'])    
        return False
    st.success(f'Expert={expert_name} has been deleted successfully')
    return True

def list_datasets(username:str=None, name_only:bool=False, include_shared:bool=False):
    # list all datasets (of specific owner)
    
    if include_shared:
        
        response = requests.get(api_urls['get_dataset'], json = {'owner':username}).json()
    else:
        
        response = requests.get(api_urls['get_owner_dataset'], json = {'owner':username}).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])    
        return []
    
    datasets = [e['dataset_name'] if e['owner'] == username else f"{e['dataset_name']}@{e['owner']}" for e in response['response'] ]
    # if name_only:
    #     return [d['name'] if d['owner'] == username else f"{d['name']}@{d['owner']}" for d in datasets]
    
    return datasets


def add_shared_users_to_expert(owner:str, expert_name:str, share_boolean:bool, shared_users:list=[]):
    if not share_boolean:
        return True
    
    
    try:
        response = requests.post(api_urls['share_expert'], json = {'owner':owner, 'expert_name':expert_name, 'shared_users':shared_users}).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
    except Exception as e:
            api_fail('Dataset sharing failed, '+ e.__str__() )
            return False
    return True

def add_shared_users_to_dataset(owner:str, dataset_name:str, share_boolean:bool, shared_users:list):

    if not share_boolean:
        return True
    
    
    try:
        response = requests.post(api_urls['share_dataset'], json = {'owner':owner, 'dataset_name':dataset_name, 'shared_users':shared_users}).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
    except Exception as e:
            api_fail('Dataset sharing failed, '+ e.__str__() )
            return False
    return True


# def create_dataset(username, dataset_name, dataset_description, uploaded_files, 
#                    share_or_not:bool, shared_user_accounts:list=[]):
#     # validate inputs
#     uploaded_file_names = [f.name for f in uploaded_files]
#     dataset = config.dataset.create(username, dataset_name, dataset_description, uploaded_file_names, shared_user_accounts)
#     ## check file size/extension is non-empty/extremely large
#     for f in uploaded_files:
#         if f.size == 0:
#             st.error(f'File={f.name} is empty')
#             return False
#     ## check dataset name is valid: not used already
#     user_datasets = list_datasets(username, name_only=True)
#     if dataset.name in user_datasets:
#         st.error(f'Dataset={dataset_name} already exists')
#         return False
#     ## if share_or_not = True, check shared_user_accounts is non-empty
#     if share_or_not:
#         if len(shared_user_accounts) == 0:
#             st.error('Please select user(s) to share dataset, or disable user-sharing.')
#             return False
    
#     # save data file(s) to local path={dataset_name}
#     # for f in uploaded_files:
#     #     with open(os.path.join(DATSEST_PATH, f.name),"wb") as file:
#     #         file.write(f.getbuffer())
    
#     # save configurations
#     uploaded_file_names = [{'filename':f.name, 'MD5':''} for f in uploaded_files]
#     dataset = config.dataset.create(username, dataset_name, dataset_description, uploaded_file_names, shared_user_accounts)
#     res = dataset.save()
#     if not res:
#         st.error('Dataset creation failed due to configuration error')
#         return False
    
#     # Akasha: update vector db
#     '''TODO: update vector db'''
    
#     return True
    





def create_dataset(dataset_name:str, dataset_description:str, uploaded_files:vars, owner:str):
    """create doc files in DOC_PATH/{owner}/{dataset_name} , and call api to create dataset config.

    Args:
        dataset_name (str): dataset name
        dataset_description (str): dataset description
        uploaded_files (vars): bytes uploaded files
        owner (str): owner name

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
    """
    # validate inputs
    suc_count = 0
    
    try:
        
        DOCS_PATH = requests.get(api_urls['get_docs_path']).json()['response']
        
        if not stu.check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not stu.check_dir(owner_path):
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
        response = requests.post(api_urls['create_dataset'],json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        #st.success(f'Dataset\'{dataset_name}\' has been created successfully')
    except Exception as e:
    
        st.error('❌ Dataset creation failed, '+ e.__str__())
        return False
    return True









def edit_dataset(dataset_name:str, new_dataset_name:str, new_description:str, uploaded_files:vars, delete_file_set:set, owner:str):
    """doing files and directory edition for update dataset.
     1. check if DOCS_PATH exist or create it
     2. check if owner directory exist or create it
     3. check if new dataset name is already exist
     4. check if old dataset name is exist
     5. rename old dataset name if we use new dataset name
     6. delete files in delete_file_set from local path={new_dataset_name}
     7. check files and save uploaded files to local path={new_dataset_name}
     8. collect params and call api to update dataset config, related chromadbs. 
     

    Args:
        dataset_name (str): old dataset name
        new_dataset_name (str): new dataset name
        new_description (str): new dataset description
        uploaded_files (vars): byte uploaded files
        delete_file_set (set): filename that need to be deleted
        owner (str): owner name

    Raises:
        Exception: can not create DOCS_PATH directory
        Exception: can not create owner directory in DOCS_PATH
        Exception: new dataset name is already exist or old dataset name is not exist
        Exception: Dataset={dataset_name} is not exist
        Exception: api response status is not success
    """
    # validate inputs
        
    # save uploaded files to local path={new_dataset_name}
    # delete files in delete_file_set from local path={new_dataset_name}
    

    try:
        DOCS_PATH = requests.get(api_urls['get_docs_path']).json()['response']
        if not stu.check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not stu.check_dir(owner_path):
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
        response = requests.post(api_urls['update_dataset'], json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        
        #st.success(f'Dataset\'{dataset_name}\' has been updated successfully')
        
        
    except Exception as e:
    
        st.error('❌ Dataset edition failed, '+ e.__str__())
        return False
    
    return True 

# def edit_dataset(username, dataset_name, new_dataset_name, new_description, 
#                  uploaded_files, delete_file_set, existing_files,
#                  share_or_not:bool, shared_user_accounts:list=[],
#                  update_expert_config=True):
#     # validate inputs
#     ## check newly-uploaded file size is non-empty/not extremely large
#     for f in uploaded_files:
#         if f.size == 0:
#             # st.session_state['update-dataset-message'] = f'File={f.name} is empty'
#             st.error(f'❌ File="{f.name}" is empty')
#             return
#         if f.name in existing_files:
#             # st.session_state['update-dataset-message'] = f'File={f.name} already exists'
#             st.error(f'❌ File="{f.name}" already exists')
#             return
#     ## check new_dataset_name is valid: not used already
#     user_datasets = list_datasets(username, name_only=True)
#     if (new_dataset_name != dataset_name) and (new_dataset_name in user_datasets):
#         st.error(f'❌ Dataset={dataset_name} already exists')
#         return
    
#     # update dataset configurations
#     origin_dataset = config.dataset.from_json(username, dataset_name)
#     if (new_dataset_name != dataset_name):
#         origin_dataset.delete()
#     origin_dataset.set_name(new_dataset_name)
#     origin_dataset.set_description(new_description)
#     origin_dataset.set_share_users(shared_user_accounts if share_or_not else [])
#     origin_dataset.set_uid()
#     origin_dataset.set_last_update()
#     existing_files = origin_dataset.files()
#     origin_dataset.remove_files([ef for ef in existing_files if ef['filename'] in delete_file_set])
#     origin_dataset.add_files([{'filename':f.name, 'MD5':getMD5(f)} for f in uploaded_files])
#     origin_dataset.save()
    
#     # update expert configurations associated with this dataset
#     if update_expert_config:
#         experts = config.expert.list()
#         if dataset_name != new_dataset_name:
        
#             for exp in experts:
#                 for ds in exp['datasets']:
#                     if (ds['owner'] == username) and (ds['name'] == dataset_name) :
#                         expert_obj = config.expert.from_json(exp['owner'], exp['name'])
#                         expert_obj.remove_dataset(username, dataset_name)
#                         expert_obj.save()
#                         break
#     # save uploaded files to local path={new_dataset_name}
#     # delete files in delete_file_set from local path={new_dataset_name}
    
#     # update vector db
#     ## remove vector db with name is in [delete_file_set]
#     ## add vector db with name is in [uploaded_files]
#     renamed_message = f'renamed to {new_dataset_name} and' if (new_dataset_name != dataset_name) else ''
#     st.success(f'Dataset={dataset_name} has been {renamed_message} updated successfully')
    
#     return

# def delete_dataset(username, dataset_name, update_expert_config=True):
#     # validate inputs
#     ## check dataset_name is valid: exists in config
#     dataset = config.dataset.from_json(username, dataset_name)
#     if dataset is None:
#         st.error(f'Dataset={dataset_name} does not exist')
#         return
#     # delete dataset in configuration
#     dataset.delete()
#     # delete files of document path with name={dataset_name}
    
#     if update_expert_config:
#         ## delete vector db with name={dataset_name}
        
#         ## change config of experts using this dataset
#         experts = config.expert.list()
#         for exp in experts:
#             for ds in exp['datasets']:
#                 if (ds['owner'] == username) and (ds['name'] == dataset_name) :
#                     expert_obj = config.expert.from_json(exp['owner'], exp['name'])
#                     expert_obj.remove_dataset(username, dataset_name)
#                     expert_obj.save()
#                     break
    
#     st.experimental_rerun()
#     return



def delete_dataset(dataset_name:str, owner:str):
    """delete doc files in DOC_PATH/{owner}/{dataset_name} , and call api to delete dataset config, related chromadbs.

    Args:
        dataset_name (str): dataset name
        owner (str): owner name

    Raises:
        Exception: can not create DOCS_PATH directory
        Exception: can not create owner directory in DOCS_PATH
        Exception: Dataset={dataset_name} is not exist
        Exception: api response status is not success
    """
    try:
        DOCS_PATH = requests.get(api_urls['get_docs_path']).json()['response']
        
        
        if not stu.check_dir(DOCS_PATH):
            raise Exception(f'can not create {DOCS_PATH} directory')
        
        owner_path = Path(DOCS_PATH) / owner
        
        if not stu.check_dir(owner_path):
            raise Exception(f'can not create {owner} directory in {DOCS_PATH}')
        
        
        save_path = owner_path / dataset_name
        if not save_path.exists():
            raise Exception(f'Dataset={dataset_name} is not exist')
        
        save_path.rmdir()
        data = {
            'dataset_name': dataset_name,
            'owner': owner
        }
        response = requests.post(api_urls['delete_dataset'],json = data).json()
        
        if response['status'] != 'success':
            raise Exception(response['response'])
        
        st.success(f'Dataset\'{dataset_name}\' has been deleted successfully')
    except Exception as e:
    
        st.error('Dataset deletion failed, '+ e.__str__())
        return
    
    return




def get_file_list_of_dataset(username:str, dataset_name:str, name_only:bool=False)->list:
    if '@' in dataset_name:
        dataset_name, username = dataset_name.split('@')
    # dataset = config.dataset.from_json(username, dataset_name)
    # files = dataset.files()
    response = requests.get(api_urls['get_filename_list'], json = {'owner':username, 'dataset_name':dataset_name}).json()
   
    
    return response['response']

def get_lastupdate_of_file_in_dataset(dataset_name:str, file_name:str, owner:str):
    
    DOCS_PATH = requests.get(api_urls['get_docs_path']).json()['response']
    file_path = os.path.join(DOCS_PATH, owner, dataset_name, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update

def delete_file_of_dataset(dataset_name, file_name):
    # delete file={file_name} from local path={dataset_name}
    return 

def get_dataset_shared_users(dataset_name, username:str=None):
    shared_users = []
    return shared_users

def get_datasets_shared_with_me(username:str, name_only:bool=False):
    shared_datasets = list_datasets(username, name_only, include_shared=True)
    return shared_datasets

def get_datasets_of_expert(username:str, datasets:list, candidate_datasets:list=None):
    dataset_names = [e['name'] if e['owner'] == username else f"{e['name']}@{e['owner']}" for e in datasets]
    
    if candidate_datasets is None:
        return dataset_names
    return [d for d in dataset_names if d in candidate_datasets]

def check_expert_use_shared_dataset(expert_datasets:list, username:str=None):
    # check if expert use shared dataset
    for d in expert_datasets:
        if d['owner'] != username:
            return True
    return False

def get_expert_refer_dataset_files(username, expert_name):
    # expert_dataset_files = {}
    # expert_dataset_files = {'dataset_name_1':[{"owner":"ccchang","name":"D1","files":["F1","F2","F3"]}],
    #                         'dataset_name_2':[{"owner":"ccchang","name":"D1","files":["F1","F2","F3"]}],
    #                         'dataset_name_3':[{"owner":"ccchang","name":"D1","files":["F1","F2","F3"]}]}
    # all_dataset = list_datasets
    # for dataset in ALL_DATASET:
    #     expert_dataset_files[dataset] = get_file_list_of_dataset(username, dataset)
    expert = config.expert.from_json(username, expert_name)
    # st.write(expert, expert.datasets())
    expert_datasets = expert.datasets()
    return expert_datasets

def get_expert_shared_users(expert):
    shared_users = []
    expert.get('shared_users')
    return shared_users

def get_experts_shared_with_me(username):
    experts = config.expert.list()
    shared_expert_name = []
    for exp in experts:
        if username in exp['shared_users']:
            shared_expert_name.append(f"{exp['name']}@{exp['owner']}") 
    return shared_expert_name

# embedding model
def get_embedding_model_of_expert(expert_name):
    embedding_model = 'embedding_model_A'
    return embedding_model

# chunk size
def get_chunksize_of_expert(expert_name):
    chunksize = 10
    return chunksize

# settings
def _save_openai_configuration(key:str):
    # check if openai api key is valid
    # save openai api key
    response = requests.get(api_urls['test_openai'], json = {'openai_key':key}).json()
    if response['status'] != 'success':
        api_fail(response['response'])
        return False
    
    st.session_state['openai_key'] = key
    return True

def _save_azure_openai_configuration(key:str, endpoint:str):
    # check if azure openai credentials are valid
    # save azure openai credentials
    response = requests.get(api_urls['test_azure'], json = {'azure_key':key, 'azure_base':endpoint}).json()
    if response['status'] != 'success':
        api_fail(response['response'])
        return False
    
    st.session_state['azure_key'] = key
    st.session_state['azure_base'] = endpoint
    return True

def save_api_configs(use_openai=False, use_azure_openai=False, openai_key=None, azure_openai_key=None, azure_openai_endpoint=None):
    # save api configs
    if use_openai:
        if not _save_openai_configuration(openai_key):
            return False
        st.success('OpenAI configurations have been saved successfully')
    if use_azure_openai:
        if not _save_azure_openai_configuration(azure_openai_key, azure_openai_endpoint):
            return False
        st.success('Azure configurations have been saved successfully')
    return True

def save_openai_to_file(owner:str, use_openai=False, use_azure_openai=False, openai_key=None, azure_openai_key=None,\
    azure_openai_endpoint=None):
    
    if not use_openai:
        openai_key = ''
    if not use_azure_openai:
        azure_openai_key = ''
        azure_openai_endpoint = ''
    data = {'owner':owner, 'openai_key':openai_key, 'azure_key':azure_openai_key,
            'azure_base':azure_openai_endpoint}
    
    
    response = requests.post(api_urls['save_openai'], json = data ).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])
        return False
    
    st.success('OpenAI configuration file has been saved successfully')
    return True


def api_fail(response):
    
    if isinstance(response,str):
        st.error(f'❌ API failed: {response}')
    elif isinstance(response,list):
        res = ''.join(response)
        st.error(f'❌ API failed: {res}')
        
    return


def check_file_selected_by_expert(datasets:list, dataset_name:str, dataset_owner:str, filename:str):
    # check if the filename is in the dataset of expert config file
    
    for ds in datasets:
        if ds['name'] == dataset_name and ds['owner'] == dataset_owner:
            if filename in ds['files']:
                return True
    return False






def get_dataset_info(owner:str, dataset_name:str)->(list,str,str):
    # get dataset info from config
    response = requests.get(api_urls['show_dataset'], json = {'owner':owner, 'dataset_name':dataset_name}).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])
        return [], '', ''
    filelist = [f['filename'] for f in response['response']['files']]
    return filelist, response['response']['description'], response['response']['last_update'] 



def get_expert_info(owner:str, expert_name:str)->(list,str,str,list):
    # get dataset info from config
    response = requests.get(api_urls['show_expert'], json = {'owner':owner, 'expert_name':expert_name}).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])
        return [], '', '', []
    shared_users = []
    if 'shared_users' in response['response']:
        shared_users = response['response']['shared_users']
    
    return response['response']['datasets'], response['response']['embedding_model'], str(response['response']['chunk_size']) , shared_users




def get_openai_config(owner:str)->dict:
         
        
    data = {
        'owner': owner,
        'openai_key': st.session_state.openai_key if st.session_state.openai_on else '',
        'azure_key': st.session_state.azure_key if st.session_state.azure_openai_on else '',
        'azure_base': st.session_state.azure_base if st.session_state.azure_openai_on else ''
    }
    response = requests.get(api_urls['choose_openai'], json=data).json()
    
    if response['status'] != 'success':
        api_fail(response['response'])
        return {}
    
    return response['response']
    