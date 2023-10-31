import streamlit as st
import pandas as pd
from typing import List, Dict

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

def ask_question(col, sys_prompt, prompt, expert_name, advanced_params:dict, auto_clean=False):
    if prompt:
        with col.chat_message("user"):
            st.markdown(prompt)
        with col.chat_message("assistant"):
            # get response from expert
            success = True
            if success:
                st.write('my random answer')
                save_last_consult_for_expert(expert_name)
    if auto_clean:
        st.session_state['question'] = ''
        
def ask_question_deep(col, layers_list:List[dict], sys_prompt, prompt, expert_name, advanced_params:dict, auto_clean=False):
    if prompt:
        with col.chat_message("user"):
            st.markdown(prompt)
        with col.chat_message("assistant"):
            success = True
            if success:
                # get response from expert with chain of thought
                st.write('my random answer from layers of questions')
                save_last_consult_for_expert(expert_name)
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

def list_experts(username:str=None):
    # list all experts (of specific username)
    all_experts = ['Expert_A', 'Expert_B', 'Expert_C']
    return all_experts

def create_expert(expert_name, expert_embedding, expert_datasets, expert_chunksize):
    # validate inputs
    ## check chunksize is valid: not extremely large
    ## check expert name is valid: not used already
    
    # save inputs to config
    return 
        
def edit_expert(expert_name, new_expert_name, 
                default_expert_embedding, new_expert_embedding, 
                default_expert_chunksize, new_expert_chunksize, 
                default_expert_datasets, new_expert_datasets):
    # validate inputs
    ## update_expert_name is valid: not used already
    ## update_chunksize is valid: not extremely large
    ## new_expert_datasets is valid: not empty
    
    # rename expert name to new_expert_name
    
    # if change embedding model or chunksize, construct new vector db 
    # update configuration of expert's vector db(s) 
    
    return

def delete_expert(expert_name):
    # delete expert from all experts in config
    return

def list_datasets(username:str=None):
    # list all datasets (of specific username)
    all_datasets = ['Data_A', 'Data_B', 'Data_C']
    return all_datasets

def create_dataset(dataset_name, dataset_description, uploaded_files):
    # validate inputs
    ## check file size/extension is non-empty/extremely large
    ## check dataset name is valid: not used already
    # save data file(s) to local path={new_dataset_name}
    if True:
        st.success(f'Dataset={dataset_name} has been created successfully')
    else:
        st.error('Dataset creation failed')
        
def edit_dataset(dataset_name, new_dataset_name, new_description, uploaded_file_list, delete_file_set):
    # validate inputs
    ## check newly-uploaded file size is non-empty/not extremely large
    ## check new_dataset_name is valid: not used already
    
    # rename dataset name to new_dataset_name
    # update description to new_description
    
    # save uploaded files to local path={new_dataset_name}
    # delete files in delete_file_set from local path={new_dataset_name}
    
    # update vector db
    ## remove vector db with name is in [delete_file_set]
    ## add vector db with name is in [uploaded_file_list]
    return

def delete_dataset(dataset_name, update_expert_config=True):
    # validate inputs
    ## check dataset_name is valid: exists in config
    
    # delete files of document path with name={dataset_name}
    
    # if update_expert_config==True:
    ## delete vector db with name={dataset_name}
    ## delete config of experts using this dataset
    return

def get_file_list_of_dataset(dataset_name):
    files = ['file-1', 'file-2', 'file-3']
    return files
def get_lastupdate_of_file_in_dataset(dataset_name, file_name):
    last_update = '2022-10-10 10:55'
    return last_update

def delete_file_of_dataset(dataset_name, file_name):
    # delete file={file_name} from local path={dataset_name}
    return 

def get_description_of_dataset(dataset_name):
    description = 'docs of specific domain'
    return description

def get_lastupdate_of_dataset(dataset_name):
    last_updates = []
    dataset_files = get_file_list_of_dataset(dataset_name)
    for file in dataset_files:
        last_updates.append(get_lastupdate_of_file_in_dataset(dataset_name, file))
    return max(last_updates)

def get_datasets_of_expert(all_dataset, expert_name):
    # filter dataset names by expert name
    
    return all_dataset

# embedding model
def get_embedding_model_of_expert(expert_name):
    embedding_model = 'embedding_model_A'
    return embedding_model

# chunk size
def get_chunksize_of_expert(expert_name):
    chunksize = 10
    return chunksize

# settings
def _save_openai_configuration(key):
    # check if openai api key is valid
    # save openai api key
    return True

def _save_azure_openai_configuration(key, endpoint):
    # check if azure openai credentials are valid
    # save azure openai credentials
    return True

def save_api_configs(use_openai=False, use_azure_openai=False, openai_key=None, azure_openai_key=None, azure_openai_endpoint=None):
    # save api configs
    if use_openai:
        if not _save_openai_configuration(openai_key):
            return False
    if use_azure_openai:
        if not _save_azure_openai_configuration(azure_openai_key, azure_openai_endpoint):
            return False
    return True