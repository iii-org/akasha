import pytest
import akasha

install_requires = [
    "pypdf",
    "langchain>=0.1.0",
    "chromadb==0.4.14",
    "openai==0.27",
    "tiktoken",
    "scikit-learn<1.3.0",
    "jieba==0.42.1",
    "llama-cpp-python==0.2.6",
    "auto-gptq==0.3.1",
    "tqdm==4.65.0",
    "docx2txt==0.8",
    "rouge==1.0.1",
    "rouge-chinese==1.0.3",
    "click",
    "tokenizers>=0.13.3",
    "streamlit==1.28.2",
    "streamlit_option_menu==0.3.6",
]


def base_search(query_embeds, docs_embeds, k: int, relevancy_threshold: float,
                logs: dict):
    from scipy.spatial.distance import euclidean

    distance = [[euclidean(query_embeds, docs_embeds[idx]), idx]
                for idx in range(len(docs_embeds))]
    distance = sorted(distance, key=lambda x: x[0])

    # print(distance) #if (1 - dist) >= relevancy_threshold
    return [idx for dist, idx in distance[:k]]


# def base_model(prompt: str):
#     import openai
#     from langchain_community.chat_models import ChatOpenAI

#     model = akasha.helper.handle_model("gpt-3.5-turbo",
#                                        False,
#                                        0.0,
#                                        max_output_tokens=1000)
#     ret = akasha.helper.call_model(model, prompt)

#     return ret


def base_embed(texts: list) -> list:
    import numpy as np

    embeds = np.array([[0.1 * 1000] for _ in range(len(texts))])

    return embeds


@pytest.fixture
def base_line():
    ak = akasha.RAG(
        embeddings="openai:text-embedding-3-small",
        model="openai:gpt-3.5-turbo",
        verbose=False,
        chunk_size=500,
        max_input_tokens=3010,
        temperature=0.15,
        system_prompt=
        "You are the expert of Market Intelligence and Consulting Institute, please answer the following questions: ",
    )
    return ak


@pytest.mark.akasha
def test_RAG(base_line: akasha.RAG):
    ak = base_line
    query = "五軸是甚麼?"
    assert ak.verbose is False
    assert ak.search_type == "auto"
    assert ak.chunk_size == 500
    assert ak.max_input_tokens == 3010
    assert ak.temperature == 0.15
    assert (
        ak.system_prompt ==
        "You are the expert of Market Intelligence and Consulting Institute, please answer the following questions: "
    )
    response = ak("./docs/mic", query)
    assert isinstance(response, str)

    return


@pytest.mark.akasha
def test_ask():
    ak = akasha.ask(
        model="openai:gpt-4o",
        keep_logs=True,
        verbose=True,
        max_input_tokens=3000,
        stream=True,
    )
    res = ak("此requirement中chromadb的版本為何?", install_requires)
    ret = ""
    for r in res:
        ret += r
        continue
    assert ak.max_input_tokens == 3000
    assert "0.4.14" in ret

    mem = akasha.MemoryManager(
        memory_name="test_memory",
        model="openai:gpt-4o",
        embeddings="openai:text-embedding-3-small",
        verbose=True,
    )
    PROMPT = "我的名字是什麼?"
    mem.add_memory("Hi, 我的名字是小宋", "Hello, 小宋! 很高興認識你。")
    history_msg = mem.search_memory(PROMPT, top_k=3)
    response = ak(PROMPT, history_messages=history_msg, stream=False)

    assert "小宋" in response

    return
