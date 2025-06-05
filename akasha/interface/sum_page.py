import streamlit as st
import sys
import akasha
from pathlib import Path
import datetime

sys.path.append("../")


def summary_page():
    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = []

    st.title("Summarize")
    st.markdown("##")
    st.markdown("##")

    response_board, para_set = st.columns([2, 1])
    with para_set:
        sum_t, sum_l = st.columns([1, 1])
        with sum_t:
            summary_type = st.selectbox(
                "Summary Type",
                ["map_reduce", "refine"],
                index=0,
                help="map_reduce is faster but may lack clarity and detail, refine is slower but offers a comprehensive understanding of the content.",
            )
        with sum_l:
            summary_len = st.number_input(
                "Summary Length",
                value=300,
                min_value=100,
                max_value=1000,
                step=10,
                help="The length of the output summary you want LLM to generate.",
            )
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

        sb1, sb2 = st.columns([1, 1])
        with sb1:
            if st.button(
                "Clear",
                type="primary",
                use_container_width=True,
                help="Clear the prompt and response.",
            ):
                st.session_state.prompt_list = []
                st.session_state.response_list = []

                st.rerun()
        with sb2:
            if st.button("Submit", type="primary", use_container_width=True):
                if uploaded_file is not None:
                    # Get the file path
                    file_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    file_name = uploaded_file.name
                    bytes_data = uploaded_file.read()
                    with open(file_date + file_name, "wb") as f:
                        f.write(bytes_data)

                    if not isinstance(st.session_state.akasha_obj, akasha.summary):
                        st.session_state.akasha_obj = akasha.summary(
                            chunk_size=st.session_state.chunksize,
                            chunk_overlap=40,
                            model=st.session_state.model,
                            language="ch",
                            verbose=True,
                            record_exp="",
                            max_input_tokens=st.session_state.max_input_tokens,
                            temperature=st.session_state.temperature,
                            env_file=st.session_state.env_path,
                        )

                    ans = st.session_state.akasha_obj(
                        file_date + file_name,
                        sum_type=summary_type,
                        sum_len=summary_len,
                        chunk_size=st.session_state.chunksize,
                        max_input_tokens=st.session_state.max_input_tokens,
                        temperature=st.session_state.temperature,
                        keep_logs=True,
                        env_file=st.session_state.env_path,
                    )

                    st.session_state.prompt_list.append("summary of " + file_name)
                    st.session_state.response_list.append(ans)
                    timesp = st.session_state.akasha_obj.timestamp_list[-1]
                    if timesp in st.session_state.akasha_obj.logs:
                        st.session_state.logs[timesp] = (
                            st.session_state.akasha_obj.logs[timesp]
                        )

                    Path(file_date + file_name).unlink()
                else:
                    st.error("Please upload a file.")

    with response_board:
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])
    return
