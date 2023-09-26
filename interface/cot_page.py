import streamlit as st
import akasha
import sys
import os
sys.path.append('../')



def get_prompts(add_b):
    """add a prompt area if user push add prompt button and return the text input list

    Args:
        add_b (_type_): add prompt button area

    Returns:
        _type_: list of text_inputs
    """
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
    """implement chain of thought ui
    """
    st.title("Chain Of Thoughts")
    st.markdown('##')
    st.markdown('##')
    
    response_board, para_set = st.columns([2,1])
    
    
            
    with para_set:
        
        st.session_state.sys_prompt = st.text_area("System_Prompt", st.session_state.sys_prompt, help="The special instruction you want to give to the model.")
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
                    
                if not isinstance(st.session_state.akasha_obj, akasha.Doc_QA):
                    st.session_state.akasha_obj =  akasha.Doc_QA(embeddings=st.session_state.embed, chunk_size=st.session_state.chunksize, \
                        model=st.session_state.model, search_type=st.session_state.search_type, topK=st.session_state.topK, threshold=st.session_state.threshold, \
                        language='ch', verbose=True, record_exp="", max_token=st.session_state.max_token, \
                        temperature=st.session_state.temperature)
                    
                ans = st.session_state.akasha_obj.chain_of_thought(st.session_state.chose_doc_path, prompts, embeddings=st.session_state.embed,\
                    chunk_size=st.session_state.chunksize, model=st.session_state.model, topK=st.session_state.topK, \
                    threshold=st.session_state.threshold, search_type=st.session_state.search_type,\
                    system_prompt=st.session_state.sys_prompt, max_token=st.session_state.max_token, temperature=st.session_state.temperature)
                
                
                
                
                st.session_state.prompt_list.extend(prompts)
                st.session_state.response_list.extend(ans)

                
                
                
    
    
    with response_board:
        
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])