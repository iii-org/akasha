import akasha
import akasha.eval as eval
import akasha.summary as summary
import akasha.prompts as prompts

query1 = "五軸是甚麼?"
query2 = [
    "西門子自有工廠如何朝工業4.0 發展",
    "詳細解釋「工業4.0 成熟度指數」發展路徑的六個成熟度",
    "根據西門子自有工廠朝工業4.0發展，探討其各項工業4.0的成熟度指標",
]
install_requires = [
    "pypdf",
    "langchain>=0.1.0",
    "chromadb==0.4.14",
    "openai==0.27",
    "tiktoken",
    "lark==1.1.7",
    "scikit-learn<1.3.0",
    "jieba==0.42.1",
    "sentence-transformers==2.2.2",
    "torch==2.0.1",
    "transformers>=4.33.4",
    "llama-cpp-python==0.2.6",
    "auto-gptq==0.3.1",
    "tqdm==4.65.0",
    "docx2txt==0.8",
    "rouge==1.0.1",
    "rouge-chinese==1.0.3",
    "bert-score==0.3.13",
    "click",
    "tokenizers>=0.13.3",
    "streamlit==1.28.2",
    "streamlit_option_menu==0.3.6",
]


## used for custom search type, use query_embeds and docs_embeds to determine relevant documents.
# you can add your own variables using log to record, for example log['dd'] = "miao"
def test_search(query_embeds, docs_embeds, k: int, relevancy_threshold: float,
                log: dict):
    from scipy.spatial.distance import euclidean
    import numpy as np

    distance = [[euclidean(query_embeds, docs_embeds[idx]), idx]
                for idx in range(len(docs_embeds))]
    distance = sorted(distance, key=lambda x: x[0])

    ## change dist if embeddings not between 0~1
    max_dist = 1
    while max_dist < distance[-1][0]:
        max_dist *= 10
        relevancy_threshold *= 10

    ## add log para
    log["dd"] = "miao"

    return [
        idx for dist, idx in distance[:k]
        if (max_dist - dist) >= relevancy_threshold
    ]


def test_model(prompt: str):
    import openai
    from langchain.chat_models import ChatOpenAI

    openai.api_type = "open_ai"
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ret = model.predict(prompt)

    return ret


def test_embed(texts: list) -> list:
    from sentence_transformers import SentenceTransformer

    mdl = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeds = mdl.encode(texts, normalize_embeddings=True)

    return embeds


def QA(doc_path="./docs/mic/"):
    qa = akasha.Doc_QA(verbose=False, search_type="svm")

    qa.get_response(
        doc_path=doc_path,
        prompt=query1,
        chunk_size=500,
        record_exp="",
        search_type=test_search,
        max_doc_len=1500,
        system_prompt="請用中文回答",
    )

    print(qa.response)

    response_list = qa.chain_of_thought(doc_path=doc_path, prompt_list=query2)
    print(response_list)

    ## ask_whole_file
    response = qa.ask_whole_file(
        file_path="docs/mic/20230317_5軸工具機因應市場訴求改變的發展態勢.pdf",
        prompt=f'''五軸是甚麼?''',
        model="openai:gpt-3.5-turbo-16k",
    )
    print(response)

    ## ask_self
    qa.ask_self(model="openai:gpt-4",
                max_doc_len=5000,
                prompt="langchain的套件版本?",
                info=install_requires)
    print(qa.response)
    return qa


### logs ###
def QA_log(doc_path="./docs/mic/"):
    # set keep_logs=True to keep logs
    qa = akasha.Doc_QA(verbose=False, search_type="svm", keep_logs=True)

    qa.get_response(
        doc_path=doc_path,
        prompt=query1,
        chunk_size=500,
        max_doc_len=1500,
    )
    print(qa.response)

    ### you can use qa.logs to get the log of each run, logs is a dictonary, for each run, the key is timestamp and the log of that run is the value.
    # use qa.timestamp_list to get the timestamp of each run, so you can get the log of each run by using it as key.
    timestamps = qa.timestamp_list
    print(qa.logs[timestamps[-1]])


### EVAL ###
### remember the question_style must match the q_file's type
def EVAL(doc_path: str = "./docs/mic/"):
    eva = eval.Model_Eval(question_style="single_choice",
                          question_type="fact",
                          verbose=True)

    b, c = eva.auto_create_questionset(
        doc_path=doc_path,
        question_num=2,
        question_style="single_choice",
        chunk_size=850,
        output_file_path="questionset/mic_single_choice.txt",
    )
    # b,c = eva.auto_create_questionset(doc_path=doc_path, question_num=2,  question_type="essay")

    print(
        eva.auto_evaluation(
            questionset_file="questionset/mic_single_choice.txt",
            doc_path=doc_path,
            verbose=True,
        ))
    # print(eva.__dict__)

    # eva.optimum_combination(q_file="questionset/mic_15.txt", doc_path=doc_path, chunk_size_list=[500]\
    #     ,search_type_list=[test_search,"svm","tfidf"])
    return eva


# eval_obj = EVAL()

#eval_obj.save_logs("./eva.json")


### SUMMARY ###
def SUM(file_name: str = "./docs/mic/20230531_智慧製造需求下之邊緣運算與新興通訊發展分析.pdf"):
    sum = summary.Summary(
        chunk_size=1000,
        chunk_overlap=40,
        model="openai:gpt-3.5-turbo",
        verbose=False,
        threshold=0.2,
        language="ch",
        record_exp="1234test",
        format_prompt="請你在回答前面加上喵",
        system_prompt="用中文回答",
        max_doc_len=1500,
        temperature=0.0,
    )

    sum.summarize_file(file_path=file_name,
                       summary_type="map_reduce",
                       summary_len=100,
                       verbose=True)
    print(sum.logs)
    return sum


# summary_obj = SUM()

# summary_obj.save_logs("./summary_logs.json")
# summary_obj.save_logs(file_type="txt")

### VISION ###
## need to use gpt-4-vision ##
# ret = akasha.openai_vision(pic_path=["C:/Users/Pictures/oruya.png"], prompt="這張圖是甚麼意思?")

# print(ret)


### JSON FORMATTER ###
def JSON():
    ak = akasha.Doc_QA(
        threshold=0.0,
        verbose=True,
    )

    response = ak.ask_whole_file(file_path="docs/resume_pool/A.docx",
                                 prompt=f'''以上是受試者的履歷，請回答該受試者的學歷、經驗、專長、年資''')

    formatted_response = akasha.helper.call_JSON_formatter(
        ak.model_obj, response, keys=["學歷", "經驗", "專長", "年資"])


### agent ###
def input_func(question: str):
    response = input(question)
    return str({"question": question, "answer": response})


def agent_example1():
    input_tool = akasha.create_tool(
        "user_question_tool",
        "This is the tool to ask user question, the only one param question is the question string that has not been answered and we want to ask user.",
        func=input_func)

    ao = akasha.agent(verbose=True,
                      tools=[
                          input_tool,
                          akasha.get_saveJSON_tool(),
                      ],
                      model="openai:gpt-3.5-turbo")
    print(
        ao("逐個詢問使用者以下問題，若所有問題都回答了，則將所有問題和回答儲存成default.json並結束。問題為:1.房間燈關了嗎? \n2. 有沒有人在家?  \n3.有哪些電器開啟?\n"
           ))


def agent_example2():

    ao = akasha.agent(tools=[
        akasha.get_wiki_tool(),
        akasha.get_saveJSON_tool(),
    ], )
    print(ao("請用中文回答李遠哲跟馬英九誰比較老?將查到的資訊和答案儲存成json檔案，檔名為AGE.json"))
    ao.save_logs("ao2.json")
