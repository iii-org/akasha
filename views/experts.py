import streamlit as st
import time
from utils import get_datasets_of_expert, create_expert, delete_expert, edit_expert
from utils import get_file_list_of_dataset, get_lastupdate_of_file_in_dataset, list_datasets
from utils import check_expert_use_shared_dataset, get_expert_info
from utils import check_dataset_is_shared, check_expert_is_shared, add_shared_users_to_expert, check_file_selected_by_expert


def experts_page(EXPERTS, EMBEDDING_MODELS, DATASETS, username, USERS):
    st.title('Knowledge Management',
             help='Manage Knowledges and their knowledge bases')
    expert_option = st.radio(
        'Option', ['My Knowledges', 'New Knowledge', 'Update Knowledge'],
        horizontal=True,
        label_visibility='collapsed')
    st.markdown('')

    if expert_option == 'My Knowledges':
        _my_experts(EXPERTS, username)

    elif expert_option == 'New Knowledge':
        _new_expert(EMBEDDING_MODELS, DATASETS, username, all_users=USERS)

    elif expert_option == 'Update Knowledge':
        _update_expert(EXPERTS,
                       EMBEDDING_MODELS,
                       DATASETS,
                       username,
                       all_users=USERS)


def _my_experts(EXPERTS, username):
    st.header('My Knowledges', divider='rainbow')
    show_shared_expert = st.toggle(
        'Shared Knowledges',
        key='show-shared-experts',
        value=False,
        disabled=len([e for e in EXPERTS if check_expert_is_shared(e)]) == 0)
    if not show_shared_expert:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]

    col_name, col_data, col_embedding, col_chunksize, col_action = st.columns(
        [2, 2, 3, 1, 2])
    col_name.subheader('Name')
    col_embedding.subheader('Embedding')
    col_data.subheader('Dataset(s)')
    col_chunksize.subheader('Chunk size')
    col_action.subheader('Action')
    st.divider()
    # dataset_obj = config.dataset.from_json(dataset_owner, dataset_name)
    for expert_name in EXPERTS:
        col_name, col_data, col_embedding, col_chunksize, col_delete = st.columns(
            [2, 2, 3, 1, 2])
        col_name.subheader(expert_name)
        if check_expert_is_shared(expert_name):
            expert_name, expert_owner = expert_name.split('@')
            disable_delete = True
        else:
            expert_owner = username
            disable_delete = False

        dataset_files, embedding_model, chunk_size, _ = get_expert_info(
            expert_owner, expert_name)

        flatten_filelist = []
        for df in dataset_files:
            expander_name = df['name'] if df[
                'owner'] == username else f'{df["name"]}@{df["owner"]}'
            with col_data.expander(expander_name):
                st.text('\n'.join(df['files']))  # change the value by config
        col_data.text(
            '\n'.join(flatten_filelist))  # change the value by config
        col_embedding.text(embedding_model)
        # col_embedding.text('openai:text-embedding-ada-002') # change the value by config

        col_chunksize.text(chunk_size)  # change value by config
        col_delete.button('Delete',
                          f'btn-delete-{expert_owner}-{expert_name}',
                          type='primary',
                          on_click=delete_expert,
                          args=(username, expert_name),
                          disabled=disable_delete)
        st.divider()


def _new_expert(EMBEDDING_MODELS, DATASETS, username, all_users):
    st.header('New Knowledge', divider='rainbow')

    st.subheader('1. Knowledge info', divider='grey')
    col_name, col_embedding, col_chunksize = st.columns([2, 3, 1])
    new_expert_name = col_name.text_input('Name',
                                          placeholder='name',
                                          help='Name of Knowledge')
    new_expert_embedding = col_embedding.selectbox(
        'Embedding Model',
        EMBEDDING_MODELS,
        index=0,
        help='Embedding model for the Knowledge')
    new_expert_chunksize = col_chunksize.number_input(
        'Chunk Size',
        min_value=10,
        value=500,
        step=1,
        help='Max number of texts to be chunked')
    st.markdown('')

    st.subheader('2. Dataset(s)', divider='grey')
    col_share, col_datasets = st.columns([1, 5])
    not_shared_datasets = list_datasets(username, name_only=True)
    shared_datasets = [d for d in DATASETS if check_dataset_is_shared(d)]

    shared = col_share.toggle('Shared Datasets',
                              value=False,
                              key='use-shared-datasets',
                              help='Use Shared Datasets',
                              disabled=len(shared_datasets) == 0)
    candidate_datasets = DATASETS if shared else not_shared_datasets
    new_expert_datas = col_datasets.multiselect(
        f'Dataset(s)',
        candidate_datasets,
        help='Select dataset(s) to be added to the Knowledge',
        label_visibility='collapsed')
    with st.expander('Select files', expanded=False):
        col_dataset, col_filename, col_file_lastupdate, col_select = st.columns(
            [2, 3, 2, 1])
        col_dataset.subheader('Dataset')
        col_filename.subheader('Filename')
        col_file_lastupdate.subheader('Last Update')
        col_select.subheader('Select')
        # if st.session_state.get('expert-add-files') is None:
        st.session_state['expert-add-files'] = dict()
        for dataset in new_expert_datas:
            st.divider()
            col_dataset, col_filename, col_file_lastupdate, col_select = st.columns(
                [2, 3, 2, 1])
            col_dataset.subheader(dataset)
            filelist = get_file_list_of_dataset(username,
                                                dataset,
                                                name_only=True)

            if st.session_state.get('expert-add-files') is None:
                st.session_state['expert-add-files'] = dict()
            odataset, owner = dataset.split('@') if check_dataset_is_shared(
                dataset) else (dataset, username)
            for filename in filelist:

                col_filename.text(filename)
                last_update = get_lastupdate_of_file_in_dataset(
                    odataset, filename, owner)
                last_update = time.ctime(last_update)
                col_file_lastupdate.text(last_update)

                if st.session_state.get(f'expert-add-files').get(
                        dataset) is None:
                    st.session_state['expert-add-files'][dataset] = set()
                checked = col_select.checkbox(
                    'select dataset file',
                    key=f'checkbox-{dataset}-{filename}',
                    value=True,
                    label_visibility='collapsed')
                if checked:
                    st.session_state['expert-add-files'][dataset].add(filename)
                else:
                    if filename in st.session_state['expert-add-files'][
                            dataset]:
                        st.session_state['expert-add-files'][dataset].remove(
                            filename)
    st.markdown('')
    # share option
    st.subheader('3. Share Option', divider='grey')
    col_share, col_share_users = st.columns([1, 3])
    other_users = [u for u in all_users if u != username]
    share_boolean = col_share.toggle('Share with other users',
                                     key=f'share-expert-{new_expert_name}',
                                     disabled=len(other_users) == 0)
    shared_users = col_share_users.multiselect(
        'User List',
        options=other_users,
        default=[],
        placeholder='Select user(s) to share',
        key=f'share-dataset-users-{new_expert_name}',
        disabled=not share_boolean,
        label_visibility='collapsed')

    # st.write(st.session_state['expert-add-files'])
    st.markdown('')
    _, col_create_expert_button, _ = st.columns([1, 2, 1])
    create_expert_button = col_create_expert_button.button(
        'Create Knowledge',
        help='Create Knowledge',
        type='primary',
        use_container_width=True)
    if create_expert_button:
        if new_expert_name == '' or new_expert_embedding == '' or new_expert_chunksize == '' or len(
                new_expert_datas) == 0:
            st.warning('Please input Knowledge information completely')
        else:
            res1 = create_expert(username, new_expert_name,
                                 new_expert_embedding, new_expert_chunksize,
                                 st.session_state['expert-add-files'])
            if res1:
                res2 = add_shared_users_to_expert(username, new_expert_name,
                                                  share_boolean, shared_users)
                if res2:
                    st.success(
                        f'Knowledge \'{new_expert_name}\' has been created successfully'
                    )


def _update_expert(EXPERTS, EMBEDDING_MODELS, DATASETS, username, all_users):
    st.header('Update Knowledge', divider='rainbow')
    EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if len(EXPERTS) == 0:
        st.warning('No Knowledge found, create one first')
    else:
        editing_expert_name = st.selectbox('Choose Knowledge', [''] + EXPERTS)
        if editing_expert_name:
            dataset_files, embedding_model, chunk_size, default_expert_shared_users = get_expert_info(
                username, editing_expert_name)
            #expert_used_datasets = [e['name'] if e['owner'] == username else f"{e['name']}@{e['owner']}" for e in dataset_files]

            # expert info
            st.subheader('1. Knowledge Info', divider='grey')
            col_name, col_embedding, col_chunksize = st.columns([2, 3, 1])
            update_expert_name = col_name.text_input('Name',
                                                     placeholder='new name',
                                                     help='Name of Knowledge',
                                                     value=editing_expert_name)

            default_expert_embedding = embedding_model
            update_expert_embedding = col_embedding.selectbox(
                'Embedding',
                options=EMBEDDING_MODELS,
                index=EMBEDDING_MODELS.index(default_expert_embedding),
                help='Embedding model of Knowledge')
            default_expert_chunksize = int(chunk_size)
            update_expert_chunksize = col_chunksize.number_input(
                'Chunk Size',
                min_value=10,
                value=default_expert_chunksize,
                help='Max number of texts to be chunked')
            # datasets
            st.subheader('2. Dataset(s)', divider='grey')
            col_share, col_datasets = st.columns([1, 5])
            not_shared_datasets = list_datasets(username,
                                                name_only=True,
                                                include_shared=False)
            shared = col_share.toggle('Shared Datasets', value=check_expert_use_shared_dataset(dataset_files, username), key='use-shared-datasets',\
                help='Use Shared Datasets', disabled=(len(DATASETS) - len(not_shared_datasets))==0)
            candidate_datasets = DATASETS if shared else not_shared_datasets

            default_expert_datasets = get_datasets_of_expert(
                username, dataset_files, candidate_datasets)
            # st.write(candidate_datasets, default_expert_datasets)
            expert_dataset_names = col_datasets.multiselect(
                'Dataset(s)',
                options=candidate_datasets,
                default=default_expert_datasets,
                help='Select dataset(s) to be added to the Knowledge',
                label_visibility='collapsed')

            with st.expander('Select files'):
                col_dataset, col_filename, col_file_lastupdate, col_select = st.columns(
                    [2, 3, 2, 1])
                col_dataset.subheader('Dataset')
                col_filename.subheader('Filename')
                col_file_lastupdate.subheader('Last Update')
                col_select.subheader('Select')
                st.session_state[
                    f'expert-add-files-{editing_expert_name}'] = dict()

                for dataset_name in expert_dataset_names:
                    st.divider()
                    dataset_session_state = dataset_name
                    if check_dataset_is_shared(dataset_name):
                        dataset_name, dataset_owner = dataset_name.split('@')
                    else:
                        dataset_owner = username

                    col_dataset, col_filename, col_file_lastupdate, col_select = st.columns(
                        [2, 3, 2, 1])
                    col_dataset.subheader(dataset_name +
                                          f'@{dataset_owner}' if dataset_owner
                                          != username else dataset_name)
                    filelist = get_file_list_of_dataset(dataset_owner,
                                                        dataset_name,
                                                        name_only=True)
                    if st.session_state.get(
                            f'expert-add-files-{editing_expert_name}') is None:
                        st.session_state[
                            f'expert-add-files-{editing_expert_name}'] = dict(
                            )

                    for filename in filelist:
                        col_filename.text(filename)
                        last_update = get_lastupdate_of_file_in_dataset(
                            dataset_name, filename, dataset_owner)
                        col_file_lastupdate.text(time.ctime(last_update))

                        if st.session_state.get(
                                f'expert-add-files-{editing_expert_name}').get(
                                    dataset_session_state) is None:
                            st.session_state[
                                f'expert-add-files-{editing_expert_name}'][
                                    dataset_session_state] = set()

                        # st.write((filename, dataset_name, dataset_owner))

                        file_used_by_expert = check_file_selected_by_expert(
                            dataset_files, dataset_name, dataset_owner,
                            filename)
                        checked = col_select.checkbox(
                            ' ',
                            key=
                            f'checkbox-{dataset_owner}-{dataset_name}-{filename}',
                            value=file_used_by_expert,
                            label_visibility='collapsed')
                        if checked:
                            st.session_state[
                                f'expert-add-files-{editing_expert_name}'][
                                    dataset_session_state].add(filename)
                        else:
                            if filename in st.session_state[
                                    f'expert-add-files-{editing_expert_name}'][
                                        dataset_session_state]:
                                st.session_state[
                                    f'expert-add-files-{editing_expert_name}'][
                                        dataset_session_state].remove(filename)
            st.markdown('')

            # share option
            st.subheader('3. Share Option', divider='grey')
            col_share, col_share_users = st.columns([1, 3])
            other_users = [u for u in all_users if u != username]

            share_boolean = col_share.toggle(
                'Share with other users',
                key=f'share-expert-{editing_expert_name}',
                value=len(default_expert_shared_users) > 0,
                disabled=len(other_users) == 0)
            shared_users = col_share_users.multiselect(
                'User List',
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
            update_expert_button = col_update.button(
                'Update Knowledge',
                f'btn-update-{editing_expert_name}',
                use_container_width=True,
                type='primary',
                help='Update Existing Knowledge')
            if update_expert_button:
                res_update = edit_expert(
                    username, editing_expert_name, update_expert_name,
                    default_expert_embedding, update_expert_embedding,
                    default_expert_chunksize, update_expert_chunksize,
                    dataset_files, st.
                    session_state[f'expert-add-files-{editing_expert_name}'],
                    share_boolean, shared_users)
                if res_update:
                    rename_msg = f' and renamed as {update_expert_name} ' if editing_expert_name != update_expert_name else ''
                    st.success(
                        f'Knowledge \'{editing_expert_name}\' has been updated{rename_msg} successfully'
                    )

            create_expert_button = col_new.button(
                'Save as New Knowledge',
                f'btn-copy-{editing_expert_name}',
                use_container_width=True,
                type='primary',
                help='Save the configuration as new Knowledge')
            if create_expert_button:

                res1 = create_expert(username, update_expert_name, update_expert_embedding, update_expert_chunksize, \
                    st.session_state[f'expert-add-files-{editing_expert_name}'])
                if res1:
                    res2 = add_shared_users_to_expert(username,
                                                      update_expert_name,
                                                      share_boolean,
                                                      shared_users)
                    if res2:
                        st.success(
                            f'Knowledge \'{update_expert_name}\' has been created successfully'
                        )
