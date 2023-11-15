import streamlit as st
import config
from utils import get_datasets_of_expert, get_expert_refer_dataset_files, create_expert, delete_expert, edit_expert
from utils import get_file_list_of_dataset, get_lastupdate_of_file_in_dataset
from utils import get_datasets_shared_with_me, check_expert_use_shared_dataset
from utils import check_dataset_is_shared, check_expert_is_shared

def experts_page(EXPERTS, EMBEDDING_MODELS, DATASETS, username, USERS):
    st.title('Expert Management', help='Manage experts and their knowledge bases')
    expert_option = st.radio('Option', ['My Experts', 'New Expert', 'Update Expert'], horizontal=True, label_visibility='collapsed')
    st.markdown('')
    if expert_option == 'My Experts':
        _my_experts(EXPERTS, username)
    elif expert_option == 'New Expert':
        _new_expert(EMBEDDING_MODELS, DATASETS, username, all_users=USERS)
            
    elif expert_option == 'Update Expert':
        _update_expert(EXPERTS, EMBEDDING_MODELS, DATASETS, username, all_users=USERS)
        
def _my_experts(EXPERTS, username):
    st.header('My Experts', divider='rainbow')
    show_shared_expert = st.toggle('Shared Experts', key='show-shared-experts', value=False, disabled=len([e for e in EXPERTS if check_expert_is_shared(e)])==0)
    if not show_shared_expert:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    # st.write(EXPERTS)
    col_name, col_data, col_embedding, col_chunksize, col_action = st.columns([3, 3, 2, 2, 2])
    col_name.subheader('Name')
    col_embedding.subheader('Embedding')
    col_data.subheader('Dataset(s)')
    col_chunksize.subheader('Chunk size')
    col_action.subheader('Action')
    st.divider()
    # dataset_obj = config.dataset.from_json(dataset_owner, dataset_name)
    for expert_name in EXPERTS:
        col_name, col_data, col_embedding, col_chunksize, col_delete = st.columns([3, 3, 2, 2, 2])
        col_name.subheader(expert_name)
        if check_expert_is_shared(expert_name):
            expert_name, expert_owner = expert_name.split('@')
            disable_delete = True
        else:
            expert_owner = username
            disable_delete = False
        
        expert_obj = config.expert.from_json(expert_owner, expert_name)
        dataset_files = expert_obj.datasets() #{} #get_expert_refer_dataset_files(username, exp)
        flatten_filelist = []
        for df in dataset_files:
            expander_name = f"{df.get('name')}@{df.get('owner')}" if df.get('owner', username) != username else df.get('name')
            with col_data.expander(expander_name):
                files = df.get('files', [])
                st.text('\n'.join(files)) # change the value by config
        col_data.text('\n'.join(flatten_filelist)) # change the value by config
        col_embedding.markdown(f"""<span style="word-wrap:break-word;">{expert_obj.embedding_model()}</span>""", unsafe_allow_html=True)
        # col_embedding.text('openai:text-embedding-ada-002') # change the value by config
        
        col_chunksize.text(expert_obj.chunk_size()) # change value by config
        col_delete.button('Delete', f'btn-delete-{expert_name}', type='primary', 
                          on_click=delete_expert, args=(username, expert_name),
                          disabled=disable_delete)
        st.divider()   
        
def _new_expert(EMBEDDING_MODELS, DATASETS, username, all_users):
    st.header('New Expert', divider='rainbow')

    st.subheader('1. Expert info', divider='grey')
    col_name, col_embedding, col_chunksize = st.columns([2, 3, 1])
    new_expert_name = col_name.text_input('Name', placeholder='name', help='Name of expert')
    new_expert_embedding = col_embedding.selectbox('Embedding Model', EMBEDDING_MODELS, index=0, help='Embedding model for the expert') 
    new_expert_chunksize = col_chunksize.number_input('Chunk Size', min_value=1, value=10, step=1, help='Max number of texts to be chunked')
    st.markdown('')
    
    st.subheader('2. Dataset(s)', divider='grey')
    col_share, col_datasets = st.columns([1, 5])
    shared_datasets = get_datasets_shared_with_me(username, name_only=True)
    shared = col_share.toggle('Shared Datasets', value=False, key='use-shared-datasets', help='Use Shared Datasets', disabled=len(shared_datasets)==0)  
    candidate_datasets = shared_datasets if shared else DATASETS
    new_expert_datas = col_datasets.multiselect(f'Dataset(s)', candidate_datasets, help='Select dataset(s) to be added to the expert', label_visibility='collapsed')
    with st.expander('Select files', expanded=False):
        col_dataset, col_filename, col_file_lastupdate, col_select = st.columns([2, 3, 2, 1])
        col_dataset.subheader('Dataset')
        col_filename.subheader('Filename')
        col_file_lastupdate.subheader('Last Update')
        col_select.subheader('Select')
        # if st.session_state.get('expert-add-files') is None:
        st.session_state['expert-add-files'] = dict()
        for dataset in new_expert_datas:
            st.divider()
            col_dataset, col_filename, col_file_lastupdate, col_select = st.columns([2, 3, 2, 1])
            col_dataset.subheader(dataset)
            filelist = get_file_list_of_dataset(username, dataset, name_only=True)
            if st.session_state.get('expert-add-files') is None:
                st.session_state['expert-add-files'] = dict()
            for filename in filelist:
                col_filename.text(filename)
                last_update = get_lastupdate_of_file_in_dataset(dataset, filename)
                col_file_lastupdate.text(last_update)
                
                if st.session_state.get(f'expert-add-files').get(dataset) is None:
                    st.session_state['expert-add-files'][dataset] = set()
                checked = col_select.checkbox('', key=f'checkbox-{dataset}-{filename}', value=True)
                if checked:
                    st.session_state['expert-add-files'][dataset].add(filename)
                else:
                    if filename in st.session_state['expert-add-files'][dataset]:
                        st.session_state['expert-add-files'][dataset].remove(filename)
    st.markdown('')
    
    # share option
    st.subheader('3. Share Option', divider='grey')
    col_share, col_share_users = st.columns([1, 3])
    other_users = [u for u in all_users if u != username]
    share_boolean = col_share.toggle('Share with other users', key=f'share-expert-{new_expert_name}', disabled=len(other_users)==0)
    shared_users = col_share_users.multiselect('User List', 
                                                options=other_users, 
                                                default=[],
                                                placeholder='Select user(s) to share',
                                                key=f'share-dataset-users-{new_expert_name}', 
                                                disabled=not share_boolean, 
                                                label_visibility='collapsed')
    
    # st.write(st.session_state['expert-add-files']) 
    st.markdown('')       
    _, col_create_expert_button, _ = st.columns([1, 2, 1])
    create_expert_button = col_create_expert_button.button('Create Expert', help='Create Expert', type='primary', use_container_width=True)
    if create_expert_button:
        res = create_expert(username, new_expert_name, new_expert_embedding, new_expert_chunksize, 
                            new_expert_datas, st.session_state['expert-add-files'],
                            share_boolean, shared_users)
        if res:
            st.success(f'Expert={new_expert_name} has been created successfully')
        
            
def _update_expert(EXPERTS, EMBEDDING_MODELS, DATASETS, username, all_users):
    st.header('Update Expert', divider='rainbow')
    EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if len(EXPERTS) == 0:
        st.warning('No expert found, create one first')
    else:
        editing_expert_name = st.selectbox('Choose Expert', [''] + EXPERTS)
        if editing_expert_name:
            editing_expert_obj = config.expert.from_json(username, editing_expert_name)
            expert_used_datasets = get_expert_refer_dataset_files(username, editing_expert_name)
            # st.write(expert_used_datasets)
            # expert info
            st.subheader('1. Expert Info', divider='grey')        
            col_name, col_embedding, col_chunksize = st.columns([2, 3, 1])
            update_expert_name = col_name.text_input('Name', placeholder='new name', help='Name of expert', value=editing_expert_obj.name())
            
            default_expert_embedding = editing_expert_obj.embedding_model()
            update_expert_embedding = col_embedding.selectbox('Embedding', options=EMBEDDING_MODELS, index=EMBEDDING_MODELS.index(default_expert_embedding), 
                                                            help='Embedding model of expert')
            default_expert_chunksize = editing_expert_obj.chunk_size()
            update_expert_chunksize = col_chunksize.number_input('Chunk Size', min_value=1, value=default_expert_chunksize, 
                                                                help='Max number of texts to be chunked')
            # datasets
            st.subheader('2. Dataset(s)', divider='grey')
            col_share, col_datasets = st.columns([1, 5])
            shared_datasets = get_datasets_shared_with_me(username, name_only=True)
            shared = col_share.toggle('Shared Datasets', value=check_expert_use_shared_dataset(editing_expert_obj.datasets(), username), key='use-shared-datasets', help='Use Shared Datasets', disabled=len(shared_datasets)==0)  
            candidate_datasets = shared_datasets if shared else DATASETS
            
            default_expert_datasets = get_datasets_of_expert(username, editing_expert_obj, candidate_datasets)
            # st.write(candidate_datasets, default_expert_datasets)
            expert_dataset_names = col_datasets.multiselect('Dataset(s)', options=candidate_datasets, 
                                                            default=default_expert_datasets,
                                                            help='Select dataset(s) to be added to the expert',
                                                            label_visibility='collapsed')
            
            with st.expander('Select files'):
                col_dataset, col_filename, col_file_lastupdate, col_select = st.columns([2, 3, 2, 1])
                col_dataset.subheader('Dataset')
                col_filename.subheader('Filename')
                col_file_lastupdate.subheader('Last Update')
                col_select.subheader('Select')
                st.session_state[f'expert-add-files-{editing_expert_name}'] = dict()
                
                for dataset_name in expert_dataset_names:
                    st.divider()
                    dataset_session_state = dataset_name
                    if check_dataset_is_shared(dataset_name):
                        dataset_name, dataset_owner = dataset_name.split('@')
                    else:
                        dataset_owner = username
                        
                    col_dataset, col_filename, col_file_lastupdate, col_select = st.columns([2, 3, 2, 1])
                    col_dataset.subheader(dataset_name + f'@{dataset_owner}' if dataset_owner != username else dataset_name)
                    filelist = get_file_list_of_dataset(dataset_owner, dataset_name, name_only=True)
                    if st.session_state.get(f'expert-add-files-{editing_expert_name}') is None:
                        st.session_state[f'expert-add-files-{editing_expert_name}'] = dict()
                
                    for filename in filelist:
                        col_filename.text(filename)
                        last_update = get_lastupdate_of_file_in_dataset(dataset_name, filename)
                        col_file_lastupdate.text(last_update)
                        
                        if st.session_state.get(f'expert-add-files-{editing_expert_name}').get(dataset_session_state) is None:
                            st.session_state[f'expert-add-files-{editing_expert_name}'][dataset_session_state] = set()
                        
                        # st.write((filename, dataset_name, dataset_owner))
                        file_used_by_expert = any([(d.get('name') == dataset_name) and (d.get('owner') == dataset_owner) and (filename in d.get('files',[])) for d in expert_used_datasets]) or (all([(d.get('name') != dataset_name) or (d.get('owner') != dataset_owner) for d in expert_used_datasets]))
                        checked = col_select.checkbox('', key=f'checkbox-{dataset_name}-{filename}', value=file_used_by_expert)
                        if checked:
                            st.session_state[f'expert-add-files-{editing_expert_name}'][dataset_session_state].add(filename)
                        else:
                            if filename in st.session_state[f'expert-add-files-{editing_expert_name}'][dataset_session_state]:
                                st.session_state[f'expert-add-files-{editing_expert_name}'][dataset_session_state].remove(filename)
            st.markdown('')
            
            # share option
            st.subheader('3. Share Option', divider='grey')
            col_share, col_share_users = st.columns([1, 3])
            other_users = [u for u in all_users if u != username]
            default_expert_shared_users = editing_expert_obj.shared_users()
            share_boolean = col_share.toggle('Share with other users', key=f'share-expert-{editing_expert_name}', value=len(default_expert_shared_users) > 0, disabled=len(other_users)==0)
            shared_users = col_share_users.multiselect('User List', 
                                                        options=other_users, 
                                                        default=default_expert_shared_users,
                                                        placeholder='Select user(s) to share',
                                                        key=f'share-dataset-users-{editing_expert_name}', 
                                                        disabled=not share_boolean, 
                                                        label_visibility='collapsed')
            # st.write(888,st.session_state[f'expert-add-files-{editing_expert_name}'])
            st.markdown('')
            st.markdown('')
            col_update, col_new = st.columns([1, 1])
            update_expert_button = col_update.button('Update Expert', f'btn-update-{editing_expert_name}', use_container_width=True, type='primary',
                                                    help='Update Existing Expert')
            if update_expert_button:
                res_update = edit_expert(username, editing_expert_name, update_expert_name, 
                                        default_expert_embedding, update_expert_embedding, 
                                        default_expert_chunksize, update_expert_chunksize, 
                                        default_expert_datasets, expert_dataset_names,
                                        st.session_state[f'expert-add-files-{editing_expert_name}'],
                                        share_boolean, shared_users)
                if res_update:
                    rename_msg = f' and renamed as {editing_expert_name} ' if editing_expert_name != update_expert_name else ''
                    st.success(f'Expert={editing_expert_name} has been updated{rename_msg}successfully') 
            
            create_expert_button = col_new.button('Save as New Expert', f'btn-copy-{editing_expert_name}', use_container_width=True, type='primary',
                                                help='Save the configuration as new expert')
            if create_expert_button:
                res_create = create_expert(username, update_expert_name, update_expert_embedding, update_expert_chunksize, 
                                        expert_dataset_names, st.session_state[f'expert-add-files-{editing_expert_name}'],
                                        share_boolean, shared_users)
                if res_create:
                    st.success(f'Expert="{update_expert_name}" has been created successfully')