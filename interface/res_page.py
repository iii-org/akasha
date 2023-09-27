import streamlit as st
import akasha
import sys
import os
sys.path.append('../')



def response_page():
    """implement get response ui
    """
    st.title("Get Response")
    st.markdown('##')
    st.markdown('##')
    
    response_board, para_set = st.columns([2,1])
    
    
            
    with para_set:
        
        st.session_state.sys_prompt = st.text_area("System Prompt", st.session_state.sys_prompt, help="The special instruction you want to give to the model.")
        prompt = st.text_area("Prompt","" , help="The prompt you want to ask the model.")
        
        sb1, sb2 = st.columns([1, 1])
        with sb1:
            if st.button("Clear", type="primary", use_container_width=True, help="Clear the prompt and response."):
                st.session_state.prompt_list = []
                st.session_state.response_list = []
                st.experimental_rerun()
        with sb2:
            if st.button("Submit", type="primary", use_container_width=True):
                if st.session_state.openai_key != "": 
                    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
                print(st.session_state.chose_doc_path)
                
                ## check if the object is created correctly ##
                if not isinstance(st.session_state.akasha_obj, akasha.Doc_QA):
                    st.session_state.akasha_obj =  akasha.Doc_QA(embeddings=st.session_state.embed, chunk_size=st.session_state.chunksize, \
                        model=st.session_state.model, search_type=st.session_state.search_type, topK=st.session_state.topK, threshold=st.session_state.threshold, \
                        language='ch', verbose=True, record_exp="", max_token=st.session_state.max_token, \
                        temperature=st.session_state.temperature)
                    
                ans = st.session_state.akasha_obj.get_response(st.session_state.chose_doc_path, prompt, embeddings=st.session_state.embed,\
                    chunk_size=st.session_state.chunksize, model=st.session_state.model, topK=st.session_state.topK, \
                    threshold=st.session_state.threshold, search_type=st.session_state.search_type,\
                    system_prompt=st.session_state.sys_prompt, max_token=st.session_state.max_token, temperature=st.session_state.temperature)
                st.session_state.prompt_list.append( prompt)
                st.session_state.response_list.append(ans)
    
    
    with response_board:
        
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])
    