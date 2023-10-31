import streamlit as st
from utils import get_datasets_of_expert, create_expert, delete_expert, edit_expert
from utils import get_embedding_model_of_expert, get_chunksize_of_expert

def experts_page(EXPERTS, EMBEDDING_MODELS, DATASETS):
    st.title('Expert Management', help='Manage experts and their knowledge bases')
    expert_option = st.radio('Option', ['My Experts', 'New Expert', 'Update Expert'], horizontal=True, label_visibility='collapsed')
    st.markdown('')
    if expert_option == 'My Experts':
        _my_experts(EXPERTS)
    elif expert_option == 'New Expert':
        _new_expert(EMBEDDING_MODELS, DATASETS)
            
    elif expert_option == 'Update Expert':
        _update_expert(EXPERTS, EMBEDDING_MODELS, DATASETS)
        
def _my_experts(EXPERTS):
    st.header('My Experts', divider='rainbow')
    col_name, col_data, col_embedding, col_chunksize, col_action = st.columns([2, 2, 3, 2, 1])
    col_name.subheader('Name')
    col_embedding.subheader('Embedding')
    col_data.subheader('Dataset(s)')
    col_chunksize.subheader('Chunk size')
    col_action.subheader('Action')
    st.divider()
    for exp in EXPERTS:
        col_name, col_data, col_embedding, col_chunksize, col_delete = st.columns([2, 2, 3, 2, 1])
        col_name.subheader(exp)
        col_data.text('Dataset-1,\nDataset-2') # change the value by config
        col_embedding.text('openai:text-embedding-ada-002') # change the value by config
        
        col_chunksize.text(1) # change value by config
        col_delete.button('Delete', f'btn-delete-{exp}', on_click=delete_expert, args=(exp,))
        st.divider()   
        
def _new_expert(EMBEDDING_MODELS, DATASETS):
    st.header('New Expert', divider='rainbow')
    with st.form('Create New Expert'):
        col_name, col_embedding, col_data, col_chunksize = st.columns([1, 2, 3, 1])
        new_expert_name = col_name.text_input('Name', placeholder='name', help='Name of expert')
        new_expert_embedding = col_embedding.selectbox('Embedding Model', EMBEDDING_MODELS, index=0, help='Embedding model for the expert') 
        new_expert_datas = col_data.multiselect(f'Dataset(s)', DATASETS, default=[], help='Select dataset(s) to be added to the expert')
        new_expert_chunksize = col_chunksize.number_input('Chunk Size', min_value=1, value=10, help='Max number of texts to be chunked')
    
        st.form_submit_button('Submit', help='Create Expert', type='secondary',
                              on_click=create_expert, 
                              args=(new_expert_name, new_expert_embedding, new_expert_datas, new_expert_chunksize))
        
        
def _update_expert(EXPERTS, EMBEDDING_MODELS, DATASETS):
    st.header('Update Expert', divider='rainbow')
    editing_expert = st.selectbox('Choose Expert', [''] + EXPERTS)
    if editing_expert:
        # show expert name, embedding model, data(s), chunk size
        st.divider()
        col_name, col_embedding, col_chunksize = st.columns([2, 3, 1])
        
        # expert info
        update_expert_name = col_name.text_input('Name', placeholder='new name', help='Name of expert', value=editing_expert)
        
        default_expert_embedding = get_embedding_model_of_expert(editing_expert)
        update_expert_embedding = col_embedding.selectbox('Embedding', options=EMBEDDING_MODELS, index=EMBEDDING_MODELS.index(default_expert_embedding), 
                                                          help='Embedding model of expert')
        default_expert_chunksize = get_chunksize_of_expert(editing_expert)
        update_expert_chunksize = col_chunksize.number_input('Chunk Size', min_value=1, value=default_expert_chunksize, 
                                                             help='Max number of texts to be chunked')
        default_expert_datasets = get_datasets_of_expert(DATASETS, editing_expert)
        expert_datasets = st.multiselect('Dataset(s)', options=DATASETS, 
                                         default=default_expert_datasets, 
                                         help='Select dataset(s) to be added to the expert')
        
        st.markdown('')
        st.markdown('')
        st.markdown('')
        col_update, col_new = st.columns([1, 1])
        col_update.button('Update Expert', f'btn-update-{editing_expert}', use_container_width=True, type='primary',
                          help='Update Existing Expert',
                          on_click=edit_expert, args=(editing_expert, update_expert_name, 
                                                      default_expert_embedding, update_expert_embedding, 
                                                      default_expert_chunksize, update_expert_chunksize, 
                                                      default_expert_datasets, expert_datasets))
        col_new.button('Save as New Expert', f'btn-copy-{editing_expert}', use_container_width=True, type='primary',
                       help='Save the configuration as new expert', on_click=create_expert, 
                       args=(update_expert_name, update_expert_embedding, expert_datasets, update_expert_chunksize)
                       )