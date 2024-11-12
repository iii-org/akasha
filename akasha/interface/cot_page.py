import streamlit as st
import akasha
import sys
import os
from interface.setting import handle_api_key

sys.path.append("../")


def get_prompts(add_b):
    """add a prompt area if user push add prompt button and return the text input list

    Args:
        add_b (_type_): add prompt button area

    Returns:
        _type_: list of text_inputs
    """
    with add_b:
        add = st.button(label="Add Prompt",
                        type="primary",
                        use_container_width=True)

    if add:
        st.session_state.n_text += 1
        st.rerun()
    text_inputs = ["" for _ in range(st.session_state.n_text)]

    for i in range(st.session_state.n_text):
        # add text inputs here

        text_inputs[i] = st.text_area(
            "Prompt" + str(i + 1),
            "",
            key=i,
            help="The prompt you want to ask the model.",
        )
        # st.text_input(label="Column Name", key=i) #Pass index as key

    return text_inputs


def cot_page():
    """implement chain of thought ui"""

    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = []

    st.title("Chain Of Thoughts")
    st.markdown("##")
    st.markdown("##")
    run_flag = True
    response_board, para_set = st.columns([2, 1])

    with para_set:
        st.session_state.sys_prompt = st.text_area(
            "System Prompt",
            st.session_state.sys_prompt,
            help="The special instruction you want to give to the model.",
        )
        text_prompt, bsd = st.columns([999, 1])
        sb1, sb2, sb3 = st.columns([1, 1, 1])
        with text_prompt:
            prompts = get_prompts(sb1)
        with sb2:
            if st.button(
                    "Clear",
                    type="primary",
                    use_container_width=True,
                    help="Clear the prompt and response.",
            ):
                st.session_state.prompt_list = []
                st.session_state.response_list = []
                st.session_state.n_text = 1
                run_flag = True
                st.rerun()

        with sb3:
            if st.button("Submit", type="primary", use_container_width=True):
                run_flag = handle_api_key()

                if run_flag:
                    # remove empty prompts
                    new_prompts = []
                    for p in prompts:
                        if p.replace(" ", "") != "":
                            new_prompts.append(p)

                    if not isinstance(st.session_state.akasha_obj,
                                      akasha.Doc_QA):
                        st.session_state.akasha_obj = akasha.Doc_QA(
                            embeddings=st.session_state.embed,
                            chunk_size=st.session_state.chunksize,
                            model=st.session_state.model,
                            search_type=st.session_state.search_type,
                            language="ch",
                            verbose=True,
                            record_exp="",
                            max_input_tokens=st.session_state.max_input_tokens,
                            temperature=st.session_state.temperature,
                        )

                    ans = st.session_state.akasha_obj.chain_of_thought(
                        st.session_state.chose_doc_path,
                        new_prompts,
                        embeddings=st.session_state.embed,
                        chunk_size=st.session_state.chunksize,
                        model=st.session_state.model,
                        search_type=st.session_state.search_type,
                        system_prompt=st.session_state.sys_prompt,
                        max_input_tokens=st.session_state.max_input_tokens,
                        temperature=st.session_state.temperature,
                        keep_logs=True,
                    )

                    st.session_state.prompt_list.extend(new_prompts)
                    st.session_state.response_list.extend(ans)
                    timesp = st.session_state.akasha_obj.timestamp_list[-1]
                    if timesp in st.session_state.akasha_obj.logs:
                        st.session_state.logs[
                            timesp] = st.session_state.akasha_obj.logs[timesp]

    if not run_flag:
        st.error("Please input your openAI api key.")
    with response_board:
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])
