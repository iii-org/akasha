import streamlit as st
from utils import add_question_layer, ask_question, ask_question_deep, get_last_consult_for_expert

def consult_page(placeholder_hint, EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS):
    st.title('Consult Expert')
    consult_strategy = st.radio('Consult Strategy', ['Regular', 'Deep'], index=0, help='Choose strategy when consulting expert', horizontal=True, label_visibility='collapsed')
    with st.sidebar:
        with placeholder_hint:
            with st.expander('Hint', expanded=False):
                sys_prompt = st.text_area('System Prompt', help='Special instruction/hint to lead answering direction.')
        
    if consult_strategy == 'Regular':
        _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt)
    elif consult_strategy == 'Deep':
        _deep_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt)

def _regular_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt):
    st.subheader('Regular Consult', divider='rainbow', help='Single Question at a time')
    col_answer, col_question = st.columns([3, 1])
    expert_to_consult = col_question.selectbox('Choose Expert', EXPERTS)
    last_consult_config_for_expert = get_last_consult_for_expert(expert_to_consult)
    prompt = col_question.text_area('Question', help='Prompt', placeholder='Ask something', height=150, key='question')
    auto_clean = col_question.toggle('Auto Clean', value=False, key='auto-clean-on-submit', help='Clean Question upon submit')
    with col_question.expander('Advanced'):
        # search type + top k + threshold + max token + embedding model(must same as vector db) + (language model, if compression)
        language_model = st.selectbox('language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert['language_model']))
        search_type = st.selectbox('search type', SEARCH_TYPES, index=SEARCH_TYPES.index(last_consult_config_for_expert['search_type']))
        top_k = st.slider('Top K', min_value=1, max_value=10, value=last_consult_config_for_expert['top_k'])
        threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert['threshold'])
        max_token = st.slider('Max Token', min_value=1, max_value=100, value=last_consult_config_for_expert['max_token'])
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert['temperature'])
        use_compression = st.toggle('Compression', value=False, key='use-compression-layers', help='Use language model to compress question')
        if use_compression:
            compression_language_model = st.selectbox('compression language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('compression_language_model', 0)))
        advanced_params = {'language_model':language_model,
                           'search_type':search_type,
                           'top_k':top_k,
                           'threshold':threshold,
                           'max_token':max_token,
                           'temperature':temperature,
                           'compression_language_model':compression_language_model if use_compression else None}
    
    submit_question = col_question.button('Submit', type='primary', use_container_width=True, 
                                          help='Submit question to expert', 
                                          on_click=ask_question, 
                                          args=(col_answer, sys_prompt, prompt, expert_to_consult, advanced_params, auto_clean))
        

def _deep_consult(EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, sys_prompt):
    st.subheader('Deep Consult', divider='rainbow', help='Divide questions into layers of sub-questions to increase precision.')
    col_answer, col_layer_area, col_layer_config = st.columns([2, 1, 1])
    with col_layer_config:
        expert_to_consult = st.selectbox('Choose Expert', EXPERTS)
        last_consult_config_for_expert = get_last_consult_for_expert(expert_to_consult)
        prompt = st.text_area('Question', help='Final Prompt based on response from previous layer(s) of question(s)', placeholder='Ask something', height=150, key='final-question')
        auto_clean = st.toggle('Auto Clean', value=False, key='auto-clean-on-submit-layers', help='Clean Question & Layers upon submit')
        with st.expander('Advanced'):
            # search type + top k + threshold + max token + embedding model(must same as vector db) + (language model, if compression)
            language_model = st.selectbox('language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert['language_model']))
            search_type = st.selectbox('search type', SEARCH_TYPES, index=SEARCH_TYPES.index(last_consult_config_for_expert['search_type']))
            top_k = st.slider('Top K', min_value=1, max_value=10, value=last_consult_config_for_expert['top_k'])
            threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert['threshold'])
            max_token = st.slider('Max Token', min_value=1, max_value=100, value=last_consult_config_for_expert['max_token'])
            temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=last_consult_config_for_expert['temperature'])
            use_compression = st.toggle('Compression', value=False, key='use-compression-layers', help='Use language model to compress question')
            if use_compression:
                compression_language_model = st.selectbox('compression language model', LANGUAGE_MODELS, index=LANGUAGE_MODELS.index(last_consult_config_for_expert.get('compression_language_model', 0)))
            advanced_params = {'language_model':language_model,
                               'search_type':search_type,
                               'top_k':top_k,
                               'threshold':threshold,
                               'max_token':max_token,
                               'temperature':temperature,
                               'compression_language_model':compression_language_model if use_compression else None}
    with col_layer_area:
        add_layer, _ = st.columns([999, 1])
        layers = add_question_layer(col_layer_area)
    
    submit_layers_of_questions = col_layer_config.button('Submit', type='primary', use_container_width=True, 
                                                         help='Submit layers of questions to expert', 
                                                         on_click=ask_question_deep, 
                                                         args=(col_answer, layers, sys_prompt, prompt, expert_to_consult, advanced_params, auto_clean))
