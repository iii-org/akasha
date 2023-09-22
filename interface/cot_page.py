import streamlit as st
import akasha
import sys
import os
sys.path.append('../')



def get_prompts(add_b):
    with add_b:
        add = st.button(label="Add Prompt",type='primary', use_container_width=True)

    if add:
        st.session_state.n_text += 1
        st.experimental_rerun()
    text_inputs = ["" for _ in range(st.session_state.n_text)]

    for i in range(st.session_state.n_text):
    #add text inputs here
    
        text_inputs[i] = st.text_area("Prompt"+str(i+1), "", key=i, help="The prompt you want to ask the model.") 
        #st.text_input(label="Column Name", key=i) #Pass index as key

    return text_inputs
    

def cot_page():
    st.title("Chain Of Thoughts")
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
        
        sys_prompt = st.text_area("System_Prompt", "", help="The special instruction you want to give to the model.")
        text_prompt, bsd = st.columns([99,1])
        sb1, sb2, sb3 = st.columns([1, 1,1])
        with text_prompt:
            prompts = get_prompts(sb1)
        with sb2:
            if st.button("Clear", type="primary", use_container_width=True, help="Clear the prompt and response."):
                st.session_state.prompt_list = []
                st.session_state.response_list = []
                st.session_state.n_text = 1
                st.experimental_rerun()
        with sb3:
            if st.button("Submit", type="primary", use_container_width=True):
                
                if st.session_state.openai_key != "": 
                    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
                ans = akasha.chain_of_thought(final_doc_path, prompts, embed, chunksize, model, False, topK, threshold, 'ch',\
                    search_type,max_token=2500, system_prompt=sys_prompt)
                st.session_state.prompt_list.extend(prompts)
                st.session_state.response_list.extend(ans)

                
                
                
    
    
    with response_board:
        
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])