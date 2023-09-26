import streamlit as st
from pathlib import Path


def setting():
    """set the arguments for the model
    
    """
    
    
      
    st.title("Setting")
    st.markdown('##')
    st.markdown('##')
    # custom_css = """
    #     <style>
    #     /* Change the background color to a light blue (you can choose your preferred color) */
    #     input[type="number"] {
    #         background-color: lightblue !important;
    #     }
    #     </style>
    #     """
    # st.markdown(custom_css, unsafe_allow_html=True)
    
    set_model_dir()
    dp, em, md = st.columns([1,2,2])
    with dp:
        doc_path = st.selectbox("Document Path", st.session_state.docs_list, index=st.session_state.select_idx[0], help="The path of the document folder.")
        st.session_state.select_idx[0] = st.session_state.docs_list.index(doc_path)   
        try:
            st.session_state.chose_doc_path = st.session_state.docs_path + '/' + doc_path
        except:
            pass
    with em:
        st.session_state.embed = st.selectbox("Embedding Model", st.session_state.embed_list, index=st.session_state.select_idx[1],\
            help="The embedding model used to embed documents.")
        st.session_state.select_idx[1] = st.session_state.embed_list.index(st.session_state.embed)
    with md:
        st.session_state.model = st.selectbox("Language Model", st.session_state.model_list, index=st.session_state.select_idx[2], \
            help="The model used to generate response.")
        st.session_state.select_idx[2] = st.session_state.model_list.index(st.session_state.model)
    
    
    cks, tpk, seat = st.columns([1,1,1])
    with cks:
        st.session_state.chunksize = st.number_input("Chunk Size", value=st.session_state.chunksize, min_value = 100, max_value = 2000, step = 100,\
            help="The size of each chunk of the document.")
    with tpk:
        st.session_state.topK = st.number_input("Top K", value=st.session_state.topK, min_value = 1, max_value = 10, step = 1,\
            help="The number of top relevant chunks to be selected from documents.")
      
    with seat:
        st.session_state.search_type = st.selectbox("Search Type", st.session_state.search_list, index=st.session_state.select_idx[3],\
            help="The search method used to select top relevant chunks.")
        st.session_state.select_idx[3] = st.session_state.search_list.index(st.session_state.search_type)
    
    thre, tem, mxt = st.columns([1,1,1])
    
    with thre:
        st.session_state.threshold = st.number_input("Threshold",value= st.session_state.threshold , min_value=0.1, max_value=0.9, step=0.05,\
            help="The threshold used to select top relevant chunks.")
    
    with tem:
        st.session_state.temperature = st.number_input("Temperature",value= st.session_state.temperature, min_value=0.0, max_value=1.0, step=0.05,\
            help="The randomness of language model.")
    
    with mxt:
        st.session_state.max_token = st.number_input("Max Token", value= st.session_state.max_token, min_value=500, step=10,\
            help="The maximum number of tokens in the reference documents that will be used as input for the LLM model.")
        
        
        
    
def set_model_dir():
    """parse all model files(gguf) and directory in the model folder 
    """
    st.session_state.model_list = ["openai:gpt-3.5-turbo", "openai:gpt-3.5-turbo-16k"]
    modes_dir = Path(st.session_state.mdl_dir)
    for dir_path in modes_dir.iterdir():
        if dir_path.is_dir():
            st.session_state.model_list.append("hf:" + st.session_state.mdl_dir + '/' + dir_path.name)    
        elif dir_path.suffix == ".gguf":
            st.session_state.model_list.append("llama-gpu:" + st.session_state.mdl_dir + '/' + dir_path.name)