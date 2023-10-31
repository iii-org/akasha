import streamlit as st
from utils import get_file_list_of_dataset, get_description_of_dataset, get_lastupdate_of_dataset, create_dataset, delete_dataset, edit_dataset
from utils import delete_file_of_dataset, get_lastupdate_of_file_in_dataset
def dataset_page(DATASETS):
    st.title('Dataset Management')
    dataset_option = st.radio('Option', ['My Datasets', 'New Dataset', 'Update Dataset'], horizontal=True, label_visibility='collapsed')
    st.markdown('')
    
    if dataset_option == 'My Datasets':
        _my_datasets(DATASETS)
            
    elif dataset_option == 'New Dataset':
        _new_dataset()
    
    elif dataset_option == 'Update Dataset':
        _update_dataset(DATASETS)
        
def _my_datasets(DATASETS):
    st.header('My Datasets', divider='rainbow')
    col_name, col_description, col_filelist, col_lastupdate, col_actions = st.columns([2, 3, 2, 2, 1])
    col_name.subheader('Name')
    col_description.subheader('Description')
    col_filelist.subheader('Data(s)')
    col_lastupdate.subheader('Last Update')
    col_actions.subheader('Action')
    st.divider()
    for dataset in DATASETS:
    
        col_name, col_description, col_filelist, col_lastupdate, col_delete = st.columns([2, 3, 2, 2, 1])
        col_name.subheader(dataset)
        dataset_filelist = get_file_list_of_dataset(dataset) # change the value by config
        dataset_description = get_description_of_dataset(dataset) # change the value by config
        dataset_lastupdate = get_lastupdate_of_dataset(dataset) # change the value by config
        col_description.text(dataset_description) 
        col_filelist.text('\n'.join(dataset_filelist)) 
        col_lastupdate.text(dataset_lastupdate)
        col_delete.button('Delete', f'btn-delete-{dataset}', 
                          on_click=delete_dataset, args=(dataset, True))
    
        st.divider()

def _new_dataset():
    st.header('New Dataset', divider='rainbow')
    with st.form('Create New Dataset'):
        
        uploaded_files = st.file_uploader('Upload Data', type=['txt', 'pdf', 'docx', 'doc'], accept_multiple_files=True, key='upload-data-create')
        col_name, col_description = st.columns([1, 3])
        new_dataset_name = col_name.text_input('Name', placeholder='name', help='Name of dataset')
        new_dataset_description = col_description.text_input('Description', placeholder='description', help='Description of dataset')
        submitted = st.form_submit_button('Submit', help='Create Dataset', type='secondary', 
                                          on_click=create_dataset, args=(new_dataset_name, new_dataset_description, uploaded_files))
        

def _update_dataset(DATASETS):
    st.header('Update Dataset', divider='rainbow')
    editing_dataset = st.selectbox('Choose Dataset', [''] + DATASETS)
    if editing_dataset:
        # show dataset name, description, file list, last update
        st.divider()
        col_name, col_description = st.columns([1, 3])
        
        # dataset info
        update_dataset_name = col_name.text_input('Name', value=editing_dataset, placeholder='new name', help='Name of dataset')
        previous_dataset_description = get_description_of_dataset(editing_dataset)
        update_dataset_description = col_description.text_input('Description', 
                                                                value=previous_dataset_description, 
                                                                placeholder='new description', help='Description of dataset')
        # remove file list
        if st.session_state.get(f'delete-files-for-{editing_dataset}') is None:
            st.session_state[f'delete-files-for-{editing_dataset}'] = set()
        with st.expander('File List', expanded=False):
            # edit file list
            col_file, col_last_update, col_action = st.columns([2, 2, 1])
            col_file.subheader('File Name')
            col_last_update.subheader('Last Update')
            col_action.subheader('Delete')
            dataset_files = get_file_list_of_dataset(editing_dataset)
            for file in dataset_files:
                st.divider()
                col_file, col_last_update, col_action = st.columns([2, 2, 1])
                col_file.text(file)
                file_lastupdate = get_lastupdate_of_file_in_dataset(editing_dataset, file)
                col_last_update.text(file_lastupdate)
                # col_action.button('Delete', f'btn-delete-{file}-for-{editing_dataset}',
                #                   on_click=delete_file_of_dataset, args=(editing_dataset, file))
                checked = col_action.checkbox('Delete', key=f'checkbox-delete-{file}-for-{editing_dataset}', label_visibility='collapsed')
                if checked:
                    st.session_state[f'delete-files-for-{editing_dataset}'].add(file)
                else:
                    if file in st.session_state[f'delete-files-for-{editing_dataset}']:
                        st.session_state[f'delete-files-for-{editing_dataset}'].remove(file)
        
        # upload new data(s)
        uploaded_files = st.file_uploader('Upload Data', type=['txt', 'pdf', 'doc', 'docx'], accept_multiple_files=True, key='upload-data-edit')
        st.markdown('')
        # update_expert = st.radio('Update Expert', ['Yes', 'No'], index=0, horizontal=True, help='Update expert after updating dataset')
        # update_expert_boolean = True if update_expert == 'Yes' else False
        update_dataset = st.button(f'Update Dataset', f'btn-update-{editing_dataset}', use_container_width=True, type='primary',
                                   on_click=edit_dataset, 
                                   args=(editing_dataset,
                                         update_dataset_name,
                                         update_dataset_description, 
                                         uploaded_files, 
                                         st.session_state.get(f'delete-files-for-{editing_dataset}', set())))
        