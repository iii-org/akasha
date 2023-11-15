import streamlit as st
from utils import add_question_layer, ask_question, ask_question_deep
from utils import check_expert_is_shared
import config

def consult_page(placeholder_hint, EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, username):
    st.title('Consult Expert')
    consult_strategy = st.radio('Consult Strategy', ['Regular', 'Deep'], index=0, help='Choose strategy when consulting expert', horizontal=True, label_visibility='collapsed')
    with st.sidebar:
        with placeholder_hint:
            with st.expander('Hint', expanded=False):
                sys_prompt = st.text_area('System Prompt', help='Special instruction/hint to lead answering direction.')
        
    if consult_strategy == 'Regular':
        _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt, username)
    elif consult_strategy == 'Deep':
        _deep_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt, username)

def _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt, username):
    st.subheader('Regular Consult', divider='rainbow', help='Single Question at a time')
    col_answer, col_question = st.columns([3, 1])
    shared = col_question.toggle('Shared Experts', value=False, key='use-shared-experts', help='Use Shared Experts')
    if not shared:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if EXPERTS == []:
        col_question.error('No experts available, create one or use shared experts')
    else:    
        expert_name = col_question.selectbox('Choose Expert', EXPERTS)
        if check_expert_is_shared(expert_name):
            expert_name, expert_owner = expert_name.split('@')
        else:
            expert_owner = username
        expert_obj = config.expert.from_json(expert_owner, expert_name)
        consultable, reason = expert_obj.consultable(return_reason=True)
        if not consultable:
            enable_msg = 'Please contact the owner of this expert to enable consultation.' if expert_owner != username else 'Please enable consultation in "Experts" settings.'
            col_question.warning(f'Expert="{expert_name}" is currently not consultable since {reason}{enable_msg}')
        else:
            last_consult_config_for_expert = expert_obj.consultation() #get_last_consult_for_expert(expert_name)
            prompt = col_question.text_area('Question', help='Prompt', placeholder='Ask something' if consultable else '', height=150, key='question')
            auto_clean = col_question.toggle('Auto Clean', value=False, key='auto-clean-on-submit', help='Clean Question upon submit')
            with col_question.expander('Advanced'):
                # search type + top k + threshold + max token + embedding model(must same as vector db) + (language model, if compression)
                language_model = st.selectbox('language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('language_model', LANGUAGE_MODELS[0])))
                search_type = st.selectbox('search type', SEARCH_TYPES, index=SEARCH_TYPES.index(last_consult_config_for_expert.get('search_type', SEARCH_TYPES[0])))
                top_k = st.slider('Top K', min_value=1, max_value=10, value=last_consult_config_for_expert.get('top_k', 5))
                threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert.get('threshold', 0.5))
                max_token = st.slider('Max Token', min_value=1, max_value=200, value=last_consult_config_for_expert.get('max_token', 100))
                temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert.get('temperature', 0.5))
                use_compression = st.toggle('Compression', value=last_consult_config_for_expert.get('use_compression', False), key='use-compression-layers', help='Use language model to compress question')
                if use_compression:
                    compression_language_model = st.selectbox('compression language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('compression_language_model', LANGUAGE_MODELS[0])))
                advanced_params = {'language_model':language_model,
                                   'search_type':search_type,
                                   'top_k':top_k,
                                   'threshold':threshold,
                                   'max_token':max_token,
                                   'temperature':temperature,
                                   'use_compression':use_compression,
                                   'compression_language_model':compression_language_model if use_compression else ''}
            
            submit_question = col_question.button('Submit', type='primary', use_container_width=True, 
                                                help='Submit question to expert', 
                                                on_click=ask_question, 
                                                args=(col_answer, username, sys_prompt, prompt, expert_owner, expert_name, advanced_params, auto_clean),
                                                disabled=prompt == '')
        

def _deep_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt, username):
    st.subheader('Deep Consult', divider='rainbow', help='Divide questions into layers of sub-questions to increase precision.')
    col_answer, col_layer_area, col_layer_config = st.columns([2, 1, 1])
    shared = col_layer_config.toggle('Shared Experts', value=False, key='use-shared-experts', help='Use Shared Experts')
    if not shared:
        EXPERTS = [e for e in EXPERTS if not check_expert_is_shared(e)]
    if EXPERTS == []:
        col_layer_config.error('No experts available, create one or use shared experts')
    else:    
    
        with col_layer_config:
            expert_name = st.selectbox('Choose Expert', EXPERTS)
            if check_expert_is_shared(expert_name):
                expert_name, expert_owner = expert_name.split('@')
            else:
                expert_owner = username
            expert_obj = config.expert.from_json(expert_owner, expert_name)
            consultable, reason = expert_obj.consultable(return_reason=True)
            if not consultable:
                enable_msg = 'Please contact the owner of this expert to enable consultation.' if expert_owner != username else 'Please enable consultation in "Experts" settings.'
                st.warning(f'Expert="{expert_name}" is currently not consultable since {reason}{enable_msg}')
            else:
                last_consult_config_for_expert = expert_obj.consultation() #get_last_consult_for_expert(expert_name)
                prompt = st.text_area('Question', help='Final Prompt based on response from previous layer(s) of question(s)', placeholder='Ask something', height=150, key='final-question')
                auto_clean = st.toggle('Auto Clean', value=False, key='auto-clean-on-submit-layers', help='Clean Question & Layers upon submit')
                with st.expander('Advanced'):
                    # search type + top k + threshold + max token + embedding model(must same as vector db) + (language model, if compression)
                    language_model = st.selectbox('language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('language_model', LANGUAGE_MODELS[0])))
                    search_type = st.selectbox('search type', SEARCH_TYPES, index=SEARCH_TYPES.index(last_consult_config_for_expert.get('search_type', SEARCH_TYPES[0])))
                    top_k = st.slider('Top K', min_value=1, max_value=10, value=last_consult_config_for_expert.get('top_k', 5))
                    threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert.get('threshold', 0.5))
                    max_token = st.slider('Max Token', min_value=1, max_value=200, value=last_consult_config_for_expert.get('max_token', 100))
                    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert.get('temperature', 0.5))
                    use_compression = st.toggle('Compression', value=last_consult_config_for_expert.get('use_compression', False), key='use-compression-layers', help='Use language model to compress question')
                    if use_compression:
                        compression_language_model = st.selectbox('compression language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('compression_language_model', LANGUAGE_MODELS[0])))
                    advanced_params = {'language_model':language_model,
                                       'search_type':search_type,
                                       'top_k':top_k,
                                       'threshold':threshold,
                                       'max_token':max_token,
                                       'temperature':temperature,
                                       'use_compression':use_compression,
                                       'compression_language_model':compression_language_model if use_compression else ''}
                
                with col_layer_area:
                    add_layer, _ = st.columns([999, 1])
                    layers = add_question_layer(col_layer_area)
                    
                submit_layers_of_questions = st.button('Submit', type='primary', use_container_width=True, 
                                                        help='Submit layers of questions to expert', 
                                                        on_click=ask_question_deep, 
                                                        args=(col_answer, layers, username, sys_prompt, prompt, expert_owner, expert_name, advanced_params, auto_clean),
                                                        disabled=prompt == '')
    
    
