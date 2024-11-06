import streamlit as st
from utils import ask_question, get_expert_info, check_consultable, ask_chat
from utils import check_expert_is_shared, get_last_consult_for_expert, save_tmp_file, check_dataset_is_shared
from utils import get_dataset_info, get_doc_file_path, ask_summary
import os


def consult_page(DATASETS, EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS,
                 PROMPT_FORMAT_TYPES, username):
    st.title('Consult Knowledge')
    consult_strategy = st.radio(
        'Consult Strategy', ['Chat', 'Regular', 'Summary'],
        index=0,
        help='Choose strategy when consulting Knowledge',
        horizontal=True,
        label_visibility='collapsed')

    if consult_strategy == 'Chat':
        _chat_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS,
                      PROMPT_FORMAT_TYPES, username)
    elif consult_strategy == 'Regular':
        _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS,
                         PROMPT_FORMAT_TYPES, username)
    elif consult_strategy == 'Summary':
        st.subheader('File Summarization',
                     divider='rainbow',
                     help='choose a document file to summarize')
        _summary(DATASETS, LANGUAGE_MODELS, username)


def _chat_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, PROMPT_FORMAT_TYPES,
                  username):
    st.subheader('Chat', divider='rainbow', help='Single Question at a time')
    col_answer, col_question = st.columns([3, 1])
    shared = col_question.toggle('Shared Knowledges',
                                 value=False,
                                 key='use-shared-experts',
                                 help='Use Shared Knowledges')
    if col_question.button(
            'Clear History',
            type='primary',
    ):
        st.session_state.history_messages = []
    if not shared:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if EXPERTS == []:
        col_question.error(
            'No Knowledges available, create one or use shared Knowledges')
    else:
        expert_name = col_question.selectbox('Choose Knowledge', EXPERTS)
        if check_expert_is_shared(expert_name):
            expert_name, expert_owner = expert_name.split('@')
        else:
            expert_owner = username

        datasets, embeddings_model, chunk_size, _ = get_expert_info(
            expert_owner, expert_name)
        chunk_size = int(chunk_size)
        consultable, reason = check_consultable(datasets, embeddings_model,
                                                chunk_size)
        if not consultable:
            enable_msg = 'Please contact the owner of this Knowledge to enable consultation.' if expert_owner != username else 'Please enable consultation in "Knowledges" settings.'
            col_question.warning(
                f'Knowledge="{expert_name}" is currently not consultable since {reason}{enable_msg}'
            )
        else:
            last_consult_config_for_expert = get_last_consult_for_expert(
                expert_owner, expert_name)

            ### use default advanced param if shared expert, else can be modified ##
            if (expert_owner == username):
                with col_question.expander('Advanced'):
                    advanced_params = get_advance_param(
                        True, last_consult_config_for_expert, LANGUAGE_MODELS,
                        SEARCH_TYPES, PROMPT_FORMAT_TYPES, datasets,
                        chunk_size, embeddings_model)
            else:
                advanced_params = get_advance_param(
                    False, last_consult_config_for_expert, LANGUAGE_MODELS,
                    SEARCH_TYPES, PROMPT_FORMAT_TYPES, datasets, chunk_size,
                    embeddings_model)

            for message in st.session_state.history_messages:
                with col_answer.chat_message(message["role"]):
                    if "doc_metadata" in message:

                        st.markdown(message["content"],
                                    help=message["doc_metadata"])
                    else:
                        st.markdown(message["content"])
            prompt = st.chat_input("Ask your question here")
            if prompt:

                response = ask_chat(username, prompt, expert_owner,
                                    expert_name, advanced_params, col_answer)


def _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS,
                     PROMPT_FORMAT_TYPES, username):
    st.subheader('Regular Consult',
                 divider='rainbow',
                 help='Single Question at a time')
    col_answer, col_question = st.columns([3, 1])
    shared = col_question.toggle('Shared Knowledges',
                                 value=False,
                                 key='use-shared-experts',
                                 help='Use Shared Knowledges')
    if not shared:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if EXPERTS == []:
        col_question.error(
            'No Knowledges available, create one or use shared Knowledges')
    else:
        expert_name = col_question.selectbox('Choose Knowledge', EXPERTS)
        if check_expert_is_shared(expert_name):
            expert_name, expert_owner = expert_name.split('@')
        else:
            expert_owner = username

        datasets, embeddings_model, chunk_size, _ = get_expert_info(
            expert_owner, expert_name)
        chunk_size = int(chunk_size)
        consultable, reason = check_consultable(datasets, embeddings_model,
                                                chunk_size)
        if not consultable:
            enable_msg = 'Please contact the owner of this Knowledge to enable consultation.' if expert_owner != username else 'Please enable consultation in "Knowledges" settings.'
            col_question.warning(
                f'Knowledge="{expert_name}" is currently not consultable since {reason}{enable_msg}'
            )
        else:
            last_consult_config_for_expert = get_last_consult_for_expert(
                expert_owner, expert_name)

            ### use default advanced param if shared expert, else can be modified ##
            if (expert_owner == username):
                with col_question.expander('Advanced'):
                    advanced_params = get_advance_param(
                        True, last_consult_config_for_expert, LANGUAGE_MODELS,
                        SEARCH_TYPES, PROMPT_FORMAT_TYPES, datasets,
                        chunk_size, embeddings_model)
            else:
                advanced_params = get_advance_param(
                    False, last_consult_config_for_expert, LANGUAGE_MODELS,
                    SEARCH_TYPES, PROMPT_FORMAT_TYPES, datasets, chunk_size,
                    embeddings_model)

            prompt = st.chat_input("Ask your question here")
            if prompt:
                with col_answer.chat_message("user"):
                    st.markdown(prompt)
                    st.session_state['ans'] = ''

                response = ask_question(username, prompt, expert_owner,
                                        expert_name, advanced_params,
                                        col_answer)
                # with col_answer.chat_message("assistant"):
                #     placeholder = st.empty()
                #     for chunk in response.iter_content(chunk_size=16):
                #         st.session_state['ans'] += chunk.decode(
                #             'utf-8', 'ignore')
                #         placeholder.markdown(st.session_state['ans'])
            # if st.session_state['que'] != '' and st.session_state['ans'] != '':
            #     with col_answer.chat_message("user"):
            #         st.markdown(st.session_state['que'])
            #     with col_answer.chat_message("assistant"):
            #         st.markdown(st.session_state['ans'])


def get_advance_param(show: bool, param: dict, LANGUAGE_MODELS: list,
                      SEARCH_TYPES: list, PROMPT_FORMAT_TYPES: list, datasets,
                      chunk_size: int, embeddings_model: str):
    # search type + top k + threshold + max token + embedding model(must same as vector db) + (language model, if compression)
    if param['language_model'] not in LANGUAGE_MODELS:
        param['language_model'] = LANGUAGE_MODELS[0]
    if show:
        system_prompt = st.text_area(
            'System Prompt',
            help='Special instruction/hint to lead answering direction.',
            value=param.get('system_prompt', ''))
        language_model = st.selectbox('language model',
                                      LANGUAGE_MODELS,
                                      index=LANGUAGE_MODELS.index(
                                          param.get('language_model',
                                                    LANGUAGE_MODELS[0])))
        prompt_format_type = st.selectbox('Prompt Format Type',
                                          PROMPT_FORMAT_TYPES,
                                          index=PROMPT_FORMAT_TYPES.index(
                                              param.get(
                                                  'prompt_format_type',
                                                  PROMPT_FORMAT_TYPES[0])))
        search_type = st.selectbox('search type',
                                   SEARCH_TYPES,
                                   index=SEARCH_TYPES.index(
                                       param.get('search_type',
                                                 SEARCH_TYPES[0])))

        max_input_tokens = st.number_input('Max Input Tokens',
                                           min_value=500,
                                           max_value=30000,
                                           value=param.get(
                                               'max_input_tokens', 3000),
                                           step=100)

        top_k = param.get('top_k', 3)

        threshold = st.slider('Threshold',
                              min_value=0.0,
                              max_value=1.0,
                              value=param.get('threshold', 0.1))

        temperature = st.slider('Temperature',
                                min_value=0.0,
                                max_value=1.0,
                                value=param.get('temperature', 0.0))
        use_compression = st.toggle(
            'Compression',
            value=param.get('use_compression', False),
            key='use-compression-layers',
            help='Use language model to compress question')
        if use_compression:
            compression_language_model = st.selectbox(
                'compression language model',
                LANGUAGE_MODELS,
                index=LANGUAGE_MODELS.index(
                    param.get('compression_language_model',
                              LANGUAGE_MODELS[0])))
    else:
        system_prompt = param['system_prompt']
        language_model = param['language_model']
        search_type = param['search_type']
        prompt_format_type = param['prompt_format_type']
        top_k = param['top_k']
        threshold = param['threshold']
        max_input_tokens = param['max_input_tokens']
        temperature = param['temperature']
        use_compression = param['use_compression']
        compression_language_model = param['compression_language_model']

    advanced_params = {
        'datasets':
        datasets,
        'system_prompt':
        system_prompt,
        'model':
        language_model,
        'search_type':
        search_type,
        'topK':
        top_k,
        'threshold':
        threshold,
        'max_input_tokens':
        max_input_tokens,
        'temperature':
        temperature,
        'use_compression':
        use_compression,
        'chunk_size':
        chunk_size,
        'embedding_model':
        embeddings_model,
        'prompt_format_type':
        prompt_format_type,
        'compression_language_model':
        compression_language_model if use_compression else ''
    }
    return advanced_params


def _summary(DATASETS, LANGUAGE_MODELS, username):

    col_answer, setting_col = st.columns([2, 1])

    with setting_col:
        ### select models ###
        language_model = st.selectbox('language model',
                                      LANGUAGE_MODELS,
                                      index=0)
        ### summary type and sumary length ###
        sum_t, sum_l = st.columns([1, 1])
        with sum_t:
            summary_type = st.selectbox(
                "Summary Type",
                ["map_reduce", "refine"],
                index=0,
                help=
                "map_reduce is faster but may lack clarity and detail, refine is slower but offers a comprehensive understanding of the content.",
            )
        with sum_l:
            summary_len = st.number_input(
                "Summary Length",
                value=500,
                min_value=100,
                max_value=1000,
                step=10,
                help=
                "The length of the output summary you want LLM to generate.",
            )
        if len(DATASETS) == 0:
            st.session_state.sum_dataset_on = False

        tmp_dataset_on = st.toggle(
            'Use Dataset File',
            value=st.session_state.sum_dataset_on,
            disabled=len(DATASETS) == 0,
        )
        if tmp_dataset_on != st.session_state.sum_dataset_on:
            st.session_state.sum_dataset_on = tmp_dataset_on
            st.rerun()

        tmp_file_name = _select_dataset_file(DATASETS, username)

        if not st.session_state.sum_dataset_on:
            uploaded_file = st.file_uploader(
                "Upload a file", type=["txt", "pdf", "docx", "md", "csv"])

            if uploaded_file is not None:
                tmp_file_name = save_tmp_file(uploaded_file, username)

    system_prompt = st.chat_input("Type instruction here (ex. 用中文回答)")
    if system_prompt and tmp_file_name == "":
        st.warning("Please select a file first.")
    elif system_prompt:
        ask_summary(system_prompt, username, tmp_file_name, language_model,
                    summary_type, summary_len)
        ### delete tmp file is using uploaded file ###
        # if not st.session_state.sum_dataset_on:
        #     os.remove(tmp_file_name)

    if st.session_state['que'] != '' and st.session_state['ans'] != '':
        with col_answer.chat_message("user"):
            st.markdown(st.session_state['que'])
        with col_answer.chat_message("assistant"):
            st.markdown(st.session_state['ans'])


def _select_dataset_file(DATASETS: list, username: str) -> str:

    if st.session_state.sum_dataset_on:
        show_shared_dataset = st.toggle(
            'Shared Datasets',
            key='show-shared-datasets',
            value=False,
            disabled=len([d for d in DATASETS
                          if check_dataset_is_shared(d)]) == 0)
        if not show_shared_dataset:
            DATASETS = [d for d in DATASETS if not check_dataset_is_shared(d)]
        dataset_name = st.selectbox('select dataset', DATASETS, index=0)

        if check_dataset_is_shared(dataset_name):
            dataset_name, dataset_owner = dataset_name.split('@')
        else:
            dataset_owner = username
        dataset_filelist, dataset_description, dataset_lastupdate, old_shared_users = get_dataset_info(
            dataset_owner, dataset_name)

        file_name = st.selectbox('select file', dataset_filelist, index=0)

        return get_doc_file_path(dataset_owner, dataset_name, file_name)
