import streamlit as st
import akasha
import sys
import os
sys.path.append('../')



def response_page():
    st.title("Get Response")
    st.markdown('##')
    st.markdown('##')
    
    response_board, para_set = st.columns([2,1])
    
    
            
    with para_set:
        doc_path = st.selectbox("Document Path", st.session_state.docs_list, index=0, help="The path of the document folder.")
        final_doc_path = st.session_state.docs_path + '/' + doc_path
        embed = st.selectbox("Embedding Model", st.session_state.embed_list, index=0, help="The embedding model used to embed documents.")
        model = st.selectbox("Model", st.session_state.model_list, index=0, help="The model used to generate response.")
        cks, tpk = st.columns([1,1])
        with cks:
            chunksize = st.number_input("Chunk Size", value=500, min_value = 100, max_value = 2000, step = 100, help="The size of each chunk of the document.")
        with tpk:
            topK = st.number_input("Top K", value=2, min_value = 1, max_value = 10, step = 1, help="The number of top relevant chunks to be selected from documents.")
        
            
        seat, thre = st.columns([1,1])
        with seat:
            search_type = st.selectbox("Search Type", st.session_state.search_list, index=0, help="The search method used to select top relevant chunks.")
        with thre:
            threshold = st.number_input("Threshold",value= 0.2, min_value=0.1, max_value=0.9, step=0.05, help="The threshold used to select top relevant chunks.")
        
        prompt = st.text_area("Prompt","" , help="The prompt you want to ask the model.")
        sys_prompt = st.text_area("System_Prompt", "", help="The special instruction you want to give to the model.")
        sb1, sb2 = st.columns([1, 1])
        with sb1:
            if st.button("Clear", type="primary", use_container_width=True, help="Clear the prompt and response history."):
                st.session_state.prompt_list = []
                st.session_state.response_list = []
                st.experimental_rerun()
        with sb2:
            if st.button("Submit", type="primary", use_container_width=True):
                if st.session_state.openai_key != "": 
                    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
                print(final_doc_path)
                ans = akasha.get_response(final_doc_path,prompt, embed, chunksize, model, False, topK, threshold, 'ch', search_type,\
                    system_prompt=sys_prompt)
                st.session_state.prompt_list.append(prompt)
                st.session_state.response_list.append(ans)
    
    
    with response_board:
        
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])
    