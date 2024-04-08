import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv


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

        if doc_path != None and doc_path != st.session_state.select_idx[0]:
            # st.session_state.select_idx[0] = st.session_state.docs_list.index(doc_path)
            st.session_state.select_idx[0] = doc_path
            try:
                st.session_state.chose_doc_path = [
                    st.session_state.docs_path + "/" + doc_p
                    for doc_p in st.session_state.select_idx[0]
                ]
            except:
                pass
            st.experimental_rerun()
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
                st.session_state.embed)

            st.experimental_rerun()
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
                st.session_state.model)
            st.experimental_rerun()

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
            st.experimental_rerun()

    with seat:
        stp = st.selectbox(
            "Search Type",
            st.session_state.search_list,
            index=st.session_state.select_idx[3],
            help="The search method used to select top relevant chunks.",
        )
        if stp != st.session_state.search_type:
            st.session_state.search_type = stp
            st.session_state.select_idx[
                3] = st.session_state.search_list.index(
                    st.session_state.search_type)
            st.experimental_rerun()

    thre, tem, mxt = st.columns([1, 1, 1])

    with thre:
        thres = st.number_input(
            "Threshold",
            value=st.session_state.threshold,
            min_value=0.0,
            step=0.05,
            help="The threshold used to select top relevant chunks.",
        )
        if thres != st.session_state.threshold:
            st.session_state.threshold = thres
            st.experimental_rerun()
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
            st.experimental_rerun()
    with mxt:
        mt = st.number_input(
            "Max Doc Length",
            value=st.session_state.max_doc_len,
            min_value=500,
            step=10,
            help=
            "The maximum number of texts in the reference documents that will be used as input for the LLM model.",
        )

        if mt != st.session_state.max_doc_len:
            st.session_state.max_doc_len = mt
            st.experimental_rerun()


def set_model_dir():
    """parse all model files(gguf) and directory in the model folder"""

    st.session_state.model_list = [
        "openai:gpt-3.5-turbo", "openai:gpt-3.5-turbo-16k"
    ]
    try:
        modes_dir = Path(st.session_state.mdl_dir)
        for dir_path in modes_dir.iterdir():
            if dir_path.is_dir():
                st.session_state.model_list.append("hf:" +
                                                   st.session_state.mdl_dir +
                                                   "/" + dir_path.name)
            elif dir_path.suffix == ".gguf":
                st.session_state.model_list.append("llama-gpu:" +
                                                   st.session_state.mdl_dir +
                                                   "/" + dir_path.name)
    except:
        print("can not find model folder!\n\n")


def handle_api_key():
    """The function try to setup OPENAI_API_KEY and OPENAI_API_BASE to environment variable. If user input both api key and base url, will use azure openai api;
    on the orther hand, if user only input api key, will use openai api. At first it check if the user input the api key and base url,
    if not, it will check if the user has set the api key and base url in .env file.

    Returns:
        bool: True or False setup api key or not, if false, won't run the model and raise error to notify user to input api key.
    """

    run_flag = True
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "OPENAI_API_TYPE" in os.environ:
        del os.environ["OPENAI_API_TYPE"]
    if "OPENAI_API_VERSION" in os.environ:
        del os.environ["OPENAI_API_VERSION"]

    load_dotenv(Path().cwd() / ".env")

    api_token = os.environ.get("OPENAI_API_KEY")

    base_token = os.environ.get("AZURE_API_BASE")
    if base_token == None:
        base_token = os.environ.get("OPENAI_API_BASE")

    if (st.session_state.embed.split(":")[0] == "openai"
            or st.session_state.model.split(":")[0] == "openai"):
        st.session_state.akasha_obj = ""

        if (st.session_state.openai_base.replace(" ", "") != ""
                and st.session_state.openai_key.replace(" ", "") != ""):
            os.environ["OPENAI_API_BASE"] = st.session_state.openai_base
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
            if "OPENAI_API_TYPE" not in os.environ:
                os.environ["OPENAI_API_TYPE"] = "azure"
            if "OPENAI_API_VERSION" not in os.environ:
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        elif st.session_state.openai_key.replace(" ", "") != "":
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
            if "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]
            if "OPENAI_API_TYPE" in os.environ:
                del os.environ["OPENAI_API_TYPE"]
            if "OPENAI_API_VERSION" in os.environ:
                del os.environ["OPENAI_API_VERSION"]

        elif base_token != None:
            api_token = os.environ.get("AZURE_API_KEY")
            if api_token == None:
                api_token = os.environ.get("OPENAI_API_KEY")
            if api_token == None:
                return False

            if "OPENAI_API_TYPE" not in os.environ:
                os.environ["OPENAI_API_TYPE"] = "azure"
            if "OPENAI_API_VERSION" not in os.environ:
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
            os.environ["OPENAI_API_BASE"] = base_token
            os.environ["OPENAI_API_KEY"] = api_token

        elif api_token != None:
            os.environ["OPENAI_API_KEY"] = api_token
            if "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]
            if "OPENAI_API_TYPE" in os.environ:
                del os.environ["OPENAI_API_TYPE"]
            if "OPENAI_API_VERSION" in os.environ:
                del os.environ["OPENAI_API_VERSION"]

        else:
            run_flag = False
    return run_flag
