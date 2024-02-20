import streamlit as st
import time
from utils import create_dataset, delete_dataset, edit_dataset
from utils import get_lastupdate_of_file_in_dataset
from utils import check_dataset_is_shared
from utils import get_dataset_info, add_shared_users_to_dataset
from utils import get_other_users_names


def dataset_page(DATASETS, username, USERS):
    st.title('Dataset Management')
    dataset_option = st.radio('Option',
                              ['My Datasets', 'New Dataset', 'Update Dataset'],
                              horizontal=True,
                              label_visibility='collapsed')
    st.markdown('')

    if dataset_option == 'My Datasets':
        _my_datasets(DATASETS, username, all_users=USERS)

    elif dataset_option == 'New Dataset':
        _new_dataset(username, all_users=USERS)

    elif dataset_option == 'Update Dataset':
        _update_dataset(DATASETS, username, all_users=USERS)


def _my_datasets(DATASETS, username, all_users):
    st.header('My Datasets', divider='rainbow')
    show_shared_dataset = st.toggle(
        'Shared Datasets',
        key='show-shared-datasets',
        value=False,
        disabled=len([d for d in DATASETS if check_dataset_is_shared(d)]) == 0)
    if not show_shared_dataset:
        DATASETS = [d for d in DATASETS if not check_dataset_is_shared(d)]
    col_name, col_description, col_filelist, col_lastupdate, col_action = st.columns(
        [2, 2, 2, 2, 1])
    col_name.subheader('Name')
    col_description.subheader('Description')
    col_filelist.subheader('Data(s)')
    col_lastupdate.subheader('Last Update')
    # col_share_option.subheader('Share')
    col_action.subheader('Action')
    st.divider()

    for dataset_name in DATASETS:

        col_name, col_description, col_filelist, col_lastupdate, col_delete = st.columns(
            [2, 2, 2, 2, 1])
        col_name.subheader(dataset_name)
        if check_dataset_is_shared(dataset_name):
            dataset_name, dataset_owner = dataset_name.split('@')
            disable_delete = True
        else:
            dataset_owner = username
            disable_delete = False
        dataset_filelist, dataset_description, dataset_lastupdate, old_shared_users = get_dataset_info(
            dataset_owner, dataset_name)

        col_description.markdown(dataset_description)
        col_filelist.text('\n'.join(dataset_filelist))
        col_lastupdate.markdown(dataset_lastupdate)
        # col_share_option.toggle('', key=f'share-dataset-{dataset}',
        #                         disabled=len(other_users) == 0, value=False)
        col_delete.button('Delete',
                          f'btn-delete{dataset_owner}-{dataset_name}',
                          type='primary',
                          on_click=delete_dataset,
                          args=(dataset_name, username),
                          disabled=disable_delete)

        st.divider()


def _new_dataset(username, all_users):
    st.header('New Dataset', divider='rainbow')

    st.subheader('1. Dataset Info', divider='grey')
    col_name, col_description = st.columns([1, 3])
    new_dataset_name = col_name.text_input('Name',
                                           placeholder='name',
                                           help='Name of dataset')
    new_dataset_description = col_description.text_input(
        'Description',
        placeholder='description',
        help='Description of dataset')

    st.subheader('2. File List', divider='grey')
    uploaded_files = st.file_uploader('Upload Data',
                                      type=['txt', 'pdf', 'docx'],
                                      accept_multiple_files=True,
                                      key='upload-data-create')

    st.subheader('3. Share Option', divider='grey')
    col_share, col_share_users = st.columns([1, 3])
    other_users_nicknames, users_mapping = get_other_users_names(
        username, all_users)

    share_boolean = col_share.toggle('Share with other users',
                                     key=f'share-dataset-{new_dataset_name}',
                                     disabled=len(other_users_nicknames) == 0)
    shared_users = col_share_users.multiselect(
        'User List',
        options=other_users_nicknames,
        default=[],
        placeholder='Select user(s) to share',
        key=f'share-dataset-users-{new_dataset_name}',
        disabled=not (share_boolean),
        label_visibility='collapsed')
    shared_users = [users_mapping[n] for n in shared_users
                    ]  # convert "username (nickname)" to "username"

    st.markdown('')
    st.markdown('')
    _, col_create, _ = st.columns([1, 2, 1])
    create_dataset_button = col_create.button('Create Dataset',
                                              help='Create Dataset',
                                              type='primary',
                                              use_container_width=True,
                                              key='btn-create-dataset')
    if create_dataset_button:
        res = create_dataset(
            new_dataset_name, new_dataset_description,
            uploaded_files, username) and add_shared_users_to_dataset(
                username, new_dataset_name, share_boolean, shared_users)

        if res:
            st.success(
                f'Dataset \'{new_dataset_name}\' has been created successfully'
            )


def _update_dataset(DATASETS, username, all_users):
    st.header('Update Dataset', divider='rainbow')
    DATASETS = [d for d in DATASETS if not check_dataset_is_shared(d)]
    if len(DATASETS) == 0:
        st.warning('No dataset found, create one first')
    else:
        editing_dataset_name = st.selectbox('Choose Dataset', [''] + DATASETS)
        if editing_dataset_name:
            # show dataset name, description, file list, last update
            # st.divider()
            dataset_filelist, previous_dataset_description, dataset_lastupdate, old_shared_users = get_dataset_info(
                username, editing_dataset_name)

            st.subheader('1. Dataset Info', divider='grey')
            col_name, col_description = st.columns([1, 3])

            # dataset info
            update_dataset_name = col_name.text_input(
                'Name',
                value=editing_dataset_name,
                placeholder='new name',
                help='Name of dataset')
            update_dataset_description = col_description.text_input(
                'Description',
                value=previous_dataset_description,
                placeholder='new description',
                help='Description of dataset')

            # dataset file list
            st.subheader('2. File List', divider='grey')

            # upload new file(s)
            uploaded_files = st.file_uploader('Upload Data',
                                              type=['txt', 'pdf', 'docx'],
                                              accept_multiple_files=True,
                                              key='upload-data-edit')

            # Existed Files
            ## remove file list
            if st.session_state.get(
                    f'delete-files-for-{editing_dataset_name}') is None:
                st.session_state[
                    f'delete-files-for-{editing_dataset_name}'] = set()
            with st.expander('Existed Files', expanded=False):
                # edit file list
                col_file, col_last_update, col_action = st.columns([2, 2, 1])
                col_file.subheader('File Name')
                col_last_update.subheader('Last Update')
                col_action.subheader('Delete')
                dataset_files = dataset_filelist
                for file in dataset_files:
                    st.divider()
                    col_file, col_last_update, col_action = st.columns(
                        [2, 2, 1])
                    col_file.text(file)
                    file_lastupdate = get_lastupdate_of_file_in_dataset(
                        editing_dataset_name, file, username)
                    file_lastupdate = time.ctime(file_lastupdate)
                    col_last_update.text(file_lastupdate)
                    # col_action.button('Delete', f'btn-delete-{file}-for-{editing_dataset}',
                    #                   on_click=delete_file_of_dataset, args=(editing_dataset, file))
                    checked = col_action.checkbox(
                        'Delete',
                        key=
                        f'checkbox-delete-{file}-for-{editing_dataset_name}',
                        label_visibility='collapsed')
                    if checked:
                        st.session_state[
                            f'delete-files-for-{editing_dataset_name}'].add(
                                file)
                    else:
                        if file in st.session_state[
                                f'delete-files-for-{editing_dataset_name}']:
                            st.session_state[
                                f'delete-files-for-{editing_dataset_name}'].remove(
                                    file)

            # share option
            st.subheader('3. Share Option', divider='grey')
            col_share, col_share_users = st.columns([1, 3])

            #other_users = [u for u in all_users if u != username]
            other_users_nicknames, users_mapping = get_other_users_names(
                username, all_users)

            share_boolean = col_share.toggle(
                'Share with other users',
                key=f'share-dataset-{editing_dataset_name}',
                disabled=len(other_users_nicknames) == 0,
                value=len(old_shared_users) > 0)
            shared_users = col_share_users.multiselect(
                'User List',
                options=other_users_nicknames,
                default=old_shared_users,
                placeholder='Select user(s) to share',
                key=f'share-dataset-users-{editing_dataset_name}',
                disabled=not share_boolean,
                label_visibility='collapsed')

            shared_users = [users_mapping[n] for n in shared_users
                            ]  # convert "username (nickname)" to "username"

            st.markdown('')
            st.markdown('')
            _, col_update, _ = st.columns([1, 2, 1])
            update_dataset_button = col_update.button(
                f'Update Dataset',
                f'btn-update-{editing_dataset_name}',
                use_container_width=True,
                type='primary')
            if update_dataset_button:
                res1 = edit_dataset(
                    editing_dataset_name, update_dataset_name,
                    update_dataset_description, uploaded_files,
                    st.session_state.get(
                        f'delete-files-for-{editing_dataset_name}',
                        set()), username)

                res2 = add_shared_users_to_dataset(username,
                                                   update_dataset_name,
                                                   share_boolean, shared_users)
                if res1 and res2:
                    st.success(
                        f'Dataset \'{update_dataset_name}\' has been updated successfully'
                    )
