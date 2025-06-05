import streamlit as st
from pathlib import Path


def setting_page():
    """set the arguments for the model"""

    st.title("Setting")
    st.markdown("##")
    st.markdown("##")
    # custom_css = """
    #     <style>
    #     /* Change the background color to a light blue (you can choose your preferred color) */
    #     input[type="number"] {
    #         background-color: lightblue !important;
    #     }
    #     </style>
    #     """
    # st.markdown(custom_css, unsafe_allow_html=True)
    if "model_list" not in st.session_state:
        set_model_dir()

    dp, em, md = st.columns([1, 2, 2])
    with dp:
        # doc_path = st.selectbox("Document Path", st.session_state.docs_list, index=st.session_state.select_idx[0], help="The path of the document folder.")
        doc_path = st.multiselect(
            "Document Path",
            st.session_state.docs_list,
            default=st.session_state.select_idx[0],
            help="The path of the document folder.",
        )

        if doc_path is not None and doc_path != st.session_state.select_idx[0]:
            # st.session_state.select_idx[0] = st.session_state.docs_list.index(doc_path)
            st.session_state.select_idx[0] = doc_path
            try:
                st.session_state.chose_doc_path = [
                    st.session_state.docs_path + "/" + doc_p
                    for doc_p in st.session_state.select_idx[0]
                ]
            except Exception:
                pass
            st.rerun()
        print(st.session_state.chose_doc_path)
    with em:
        embed = st.selectbox(
            "Embedding Model",
            st.session_state.embed_list,
            index=st.session_state.select_idx[1],
            help="The embedding model used to embed documents.",
        )

        if embed != st.session_state.embed:
            st.session_state.embed = embed
            st.session_state.select_idx[1] = st.session_state.embed_list.index(
                st.session_state.embed
            )

            st.rerun()
    with md:
        md = st.selectbox(
            "Language Model",
            st.session_state.model_list,
            index=st.session_state.select_idx[2],
            help="The model used to generate response.",
        )
        if md != st.session_state.model:
            st.session_state.model = md
            st.session_state.select_idx[2] = st.session_state.model_list.index(
                st.session_state.model
            )
            st.rerun()

    cks, seat = st.columns([1, 1])
    with cks:
        ck = st.number_input(
            "Chunk Size",
            value=st.session_state.chunksize,
            min_value=100,
            max_value=2000,
            step=100,
            help="The size of each chunk of the document.",
        )
        if ck != st.session_state.chunksize:
            st.session_state.chunksize = ck
            st.rerun()

    with seat:
        stp = st.selectbox(
            "Search Type",
            st.session_state.search_list,
            index=st.session_state.select_idx[3],
            help="The search method used to select top relevant chunks.",
        )
        if stp != st.session_state.search_type:
            st.session_state.search_type = stp
            st.session_state.select_idx[3] = st.session_state.search_list.index(
                st.session_state.search_type
            )
            st.rerun()

    tem, mxt = st.columns([1, 2])

    with tem:
        tem = st.number_input(
            "Temperature",
            value=st.session_state.temperature,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="The randomness of language model.",
        )
        if tem != st.session_state.temperature:
            st.session_state.temperature = tem
            st.rerun()
    with mxt:
        mt = st.number_input(
            "Max Input Tokens",
            value=st.session_state.max_input_tokens,
            min_value=1000,
            step=50,
            help="The maximum number of tokens in the reference documents that will be used as input for the LLM model.",
        )

        if mt != st.session_state.max_input_tokens:
            st.session_state.max_input_tokens = mt
            st.rerun()


def set_model_dir():
    """parse all model files(gguf) and directory in the model folder"""

    st.session_state.model_list = [
        "openai:gpt-3.5-turbo",
        "openai:gpt-4o",
        "gemini:gemini-1.5-flash",
        "anthropic:claude-3-5-sonnet-20241022",
    ]
    try:
        modes_dir = Path(st.session_state.mdl_dir)
        for dir_path in modes_dir.iterdir():
            if dir_path.is_dir():
                st.session_state.model_list.append(
                    "hf:" + st.session_state.mdl_dir + "/" + dir_path.name
                )
            elif dir_path.suffix == ".gguf":
                st.session_state.model_list.append(
                    "llama-gpu:" + st.session_state.mdl_dir + "/" + dir_path.name
                )
    except Exception:
        print("can not find model folder!\n\n")
