# akasha
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/akasha-terminal?style=flat)](https://pypi.org/project/akasha-terminal/)
[![Downloads](https://img.shields.io/pypi/dm/akasha-terminal?style=flat&color=brightgreen)](https://pypi.org/project/akasha-terminal/)
[![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue?style=flat)](https://www.python.org/downloads/release/python-380/)
[![GitLab CI](https://img.shields.io/badge/GitLab%20CI-passing-orange?style=flat&logo=gitlab&logoColor=white)](https://gitlab.com)

<br/>
<br/>

# Quick Navigation

- [Introduction](#introduction)
- [Updates](#updates)
  - [Version 0.8.63](#version-0863)
  - [Version 0.8.56](#version-0856)
  - [Version 0.8.53](#version-0853)
- [Installation](#installation)
- [API Keys](#api-keys)
  - [OpenAI](#openai)
  - [Azure OpenAI](#azure-openai)
- [Usage Examples](#usage-examples)
- [Available Models](#available-models)
- [File Summarization](#file-summarization)
- [Agent Usage](#agent-usage)
- [Akasha UI](#akasha-ui)


<br/>
<br/>

# introduction

Akasha is a tool that provides two high-level features:

1. Document-based Question Answering (QA) and Retrieval Augmented Generation (RAG) using advanced Large Language Models (LLMs).
2. Flexibility to choose from a variety of language models, embedding models, and search methods for optimized results.

You can leverage popular language models and embeddings to extend Akasha's capabilities, making it adaptable to your specific needs.

For more information and to get started, please visit our Chinese manual [here](https://tea9297.github.io/) or [Chinese version of Readme](https://github.com/iii-org/akasha/blob/master/readmeTW.md).

<br/>
<br/>

# Updates

- 0.8.77

    1. Added support for Anthropic language models and vision models
    2. Deprecated the thoreshold parameter
    3. Added the function get_relevant_documents_and_scores to the retriever, which outputs documents and their similarity scores with the query
    4. self_query.query_filter metadata_field_info now accepts dictionary-type input

- 0.8.73

    1. Added the env_file parameter, allowing specification of .env file names for exporting API keys
    2. Updated the parse function in self-query; added the custom_parser parameter for custom parser functions

- 0.8.70

    1. Added support for Gemini language models and embedding models
    2. Introduced the self_query module
    3. Replaced max_doc_len with max_input_tokens and added the helper function Tokenizer.compute_tokens to calculate token length

- 0.8.66

    1. Enabled GPU inference for llama-cpp
    2. remote now supports compatibility with vllm serve

- 0.8.63

    1. add ask_image function to Doc_QA, which can use image as prompt to ask question.
    2. add max_output_tokens parameter.
    3. remote compatible with vllm serve.

- 0.8.56

    1. add extract_db_by_file and extract_db_by_keyword functions.

- 0.8.53

    1. akasha Doc_QA add stream parameter, if stream=True, return will be generator.
    2. get_response and ask_self add history_messages parameter, use to pass chat history to model
    3. prompt_format_type add chat_gpt and chat_mistral, which change prompt, system_prompt and history_messages into instruct format: [{"role":current_role, "content":prompt}]
    4. add helper function call_model, call_batch_model, call_stream_model
    5. allow pass model object (LLM) and embedding object (Embeddings) to Doc_QA, Eval, and Summary, prevent redefine of the objects.
    6. add helper function self_rag
    7. Doc_QA ask_self and ask_whole_file can allow document(info) larger then max_doc_len by calling models multiple times.
    8. Doc_QA get_response can pass dbs object to doc_path, prevent reload the chromadb data.
    9. add fastapi in akasha, you can activate  fastapi using "akasha api (--port port --host host --workers num_of_workers)"


<br/>
<br/>

<br/>
<br/>



# Installation

We recommend using Python 3.9 to run our akasha package. You can use Anaconda to create virtual environment.

```bash

# create environment

conda create --name py3-9 python=3.9
activate py3-9

# install akasha
pip install akasha-terminal


```


<br/>
<br/>

## API Keys

### OPENAI

If you want to use OpenAI models or embeddings, visit [OpenAI's API key page](https://platform.openai.com/account/api-keys) to obtain your API key.

You can save the key as **`OPENAI_API_KEY=your_api_key`** in a **`.env`** file in your current working directory, or set it as an environment variable using **`export`** in Bash, or **`os.environ`** in Python.
```bash

# set a environment variable

export OPENAI_API_KEY="your api key"


```

<br/>
<br/>

### AZURE OPENAI
If you want to use Azure OpenAI, go to [Azure OpenAI Portal](https://oai.azure.com/portal) to get your Language API base URL and key.

Before using Azure OpenAI models, you need to request access the models you want to use in [Azure OpenAI Studio](https://oai.azure.com/portal).. The deployment name should match the model name. Save the following information in a `.env` file in your current working directory:

- **`OPENAI_API_KEY=your_azure_key`**
- **`OPENAI_API_BASE=your_language_api_base_url`**
- **`OPENAI_API_TYPE=azure`**
- **`OPENAI_API_VERSION=2023-05-15`**

<br/>

If you want to save both openai key and azure key at the same time, you can also use **AZURE_API_KEY**, **AZURE_API_BASE**, **AZURE_API_TYPE**, **AZURE_API_VERSION**


```sh
## .env file
AZURE_API_KEY={your azure key}
AZURE_API_BASE={your Language API base url}
AZURE_API_TYPE=azure
AZURE_API_VERSION=2023-05-15

```


<br/>
<br/>

And now we can run akasha in python

```python
#PYTHON3.9
import akasha
ak = akasha.Doc_QA(model="openai:gpt-4")
response = ak.get_response(dir_path, prompt)


```


<br/>
<br/>




# Example

Make sure to update the `dir_path` and `prompt` variables in your code:

- **`dir_path`**: Change `"doc/"` to the folder containing the relevant PDF files with the answers.
- **`prompt`**: Set this to the specific question you want to ask.

For example:

```python
# Update dir_path and prompt accordingly
dir_path = "path/to/your/pdf_folder"
prompt = "What is the answer to my question?"
```


``` python
import akasha
import os


os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?	「失之交臂的感情」	「赤誠的心」	「浪子的真情」	「無法挽回的愛」"
ak = akasha.Doc_QA(search_type="auto")
response = ak.get_response(dir_path, prompt)
print(response)
```


``` python
「塞西莉亞花」的花語為「浪子的真情」
```

<br/>
<br/>
<br/>
<br/>


## Some models available for use

Please note that for OpenAI models, you need to set the environment variable 'OPENAI_API_KEY',
and for most Hugging Face models, a GPU is required to run them. However, .gguf models can run on CPUs.

```python
# OpenAI Models (require environment variable 'OPENAI_API_KEY' or 'AZURE_API_KEY')
openai_model = "openai:gpt-3.5-turbo"            # GPT-3.5 
openai4_model = "openai:gpt-4"                    # GPT-4 
# OpenAI Models (require environment variable 'OPENAI_API_KEY')
openai4o_model = "openai:gpt-4o"                  # GPT-4o 

# Hugging Face Models (To download the models listed below, the environment variable HUGGINGFACEHUB_API_TOKEN is required:)
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf"  # Meta Llama-2 model 

# Quantized Models (require GPU for optimal performance)
quantized_ch_llama_model = "gptq:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"  # 4-bit quantized Llama2 Chinese model
taiwan_llama_gptq = "gptq:weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"         # 4-bit quantized Taiwan LLaMa model

# Additional Hugging Face Models
mistral = "hf:Mistral-7B-Instruct-v0.2"                               # Mistral 7B instruct model
mediatek_Breeze = "hf:MediaTek-Research/Breeze-7B-Instruct-64k-v0.1"  # MediaTek Breeze model

# Llama-cpp Models (can run on CPU or GPU, require .gguf files from Hugging Face)
# Download .gguf models from https://huggingface.co/TheBloke/
# Specify the downloaded path using "llama-gpu:" or "llama-cpu:" prefix
llama_cpp_model_gpu = "llama-gpu:model/llama-2-13b-chat-hf.Q5_K_S.gguf"       # Run on GPU
llama_cpp_model_cpu = "llama-cpu:model/llama-2-7b-chat.Q5_K_S.gguf"           # Run on CPU

# Chinese Alpaca Models for Llama-cpp
llama_cpp_chinese_alpaca_gpu = "llama-gpu:model/chinese-alpaca-2-7b.Q5_K_S.gguf"  # Run on GPU
llama_cpp_chinese_alpaca_cpu = "llama-cpu:model/chinese-alpaca-2-13b.Q5_K_M.gguf"  # Run on CPU

# ChatGLM Model
chatglm_model = "chatglm:THUDM/chatglm2-6b"  # THUDM ChatGLM2 model
```


<br/>



## Some embeddings available for use

Each embedding model has a different maximum sequence length. If the text exceeds this limit, it will be cut off, which means some parts won't be represented by the model.

*rerank_base* and *rerank_large* are not embedding models. Instead, they compare the query with chunks of documents and return similarity scores. This makes them more accurate than embedding models, but also slower.

Here are the details of the different embedding and reranking models:

```python
openai_emd = "openai:text-embedding-ada-002"  # Needs "OPENAI_API_KEY"; 8192 max sequence length
huggingface_emd = "hf:all-MiniLM-L6-v2"
text2vec_ch_emd = "hf:shibing624/text2vec-base-chinese"  # 128 max sequence length
text2vec_mul_emd = "hf:shibing624/text2vec-base-multilingual"  # 256 max sequence length
text2vec_ch_para_emd = "hf:shibing624/text2vec-base-chinese-paraphrase"  # Better for long text; 256 max sequence length
bge_en_emd = "hf:BAAI/bge-base-en-v1.5"  # 512 max sequence length
bge_ch_emd = "hf:BAAI/bge-base-zh-v1.5"  # 512 max sequence length

rerank_base = "rerank:BAAI/bge-reranker-base"  # 512 max sequence length
rerank_large = "rerank:BAAI/bge-reranker-large"  # 512 max sequence length
```


<br/>
<br/>
<br/>
<br/>






## File Summarization
To create a summary of a text file in various formats like .pdf, .txt, or .docx, you can use the **Summary.summarize_file** function. For example, the following code employs the **map_reduce** summary method to instruct LLM to generate a summary of approximately 500 words.

There're two summary type, **map_reduce** and **refine**, **map_reduce** will summarize every text chunks and then use all summarized text chunks to generate a final summary; **refine** will summarize each text chunk at a time and using the previous summary as a prompt for summarizing the next segment to get a higher level of summary consistency.

```python

import akasha
sum = akasha.Summary( chunk_size=1000, chunk_overlap=100,model="openai:gpt-4")
sum.summarize_file(file_path="doc/mic/5軸工具機因應市場訴求改變的發展態勢.pdf",summary_type="map_reduce", summary_len=500\
, chunk_overlap=40)



```

<br/>
<br/>

```python
"""
### Arguments of Summary class ###
 Args:
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.
            **chunk_overlap (int, optional)**: chunk overlap of texts from documents. Defaults to 40.
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".
            **verbose (bool, optional)**: show log texts or not. Defaults to False.
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and setrecord_exp as experiment name.  default "".
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".
            **max_doc_len(int, optional)**: max docuemnt length of llm input. Defaults to 3000.
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.
"""
```

<br/>
<br/>
<br/>
<br/>


## agent
By implementing an agent, you empower the LLM with the capability to utilize tools more effectively to accomplish tasks. You can allocate tools for tasks such as file editing, conducting Google searches, and enlisting the LLM's assistance in task execution, rather than solely relying on it to respond your questions.


<br/>
<br/>
In the example1, we create a tool that can collect user inputs. Additionally, we integrate a tool into the agent's functionality to store text data in a JSON file. Following the creation of the agent, we instruct it to prompt users with questions and save their responses into a file named default.json.


```python

def input_func(question: str):
    response = input(question)
    return str({"question": question, "answer": response})



input_tool = akasha.create_tool(
    "user_question_tool",
    "This is the tool to ask user question, the only one param question is the question string that has not been answered and we want to ask user.",
    func=input_func)

ao = akasha.test_agent(verbose=True,
                    tools=[
                        input_tool,
                        akasha.get_saveJSON_tool(),
                    ],
                    model="openai:gpt-4")
print(
    ao("逐個詢問使用者以下問題，若所有問題都回答了，則將所有問題和回答儲存成default.json並結束。問題為:1.房間燈關了嗎? \n2. 有沒有人在家?  \n3.有哪些電器開啟?\n"
        ))

```

```text

I have successfully saved all the questions and answers into the "default.json" file. The conversation is now complete.

### default.json ###
[
    {
        "question": "房間燈關了嗎?",
        "answer": "no"
    },
    {
        "question": "有沒有人在家?",
        "answer": "no"
    },
    {
        "question": "有哪些電器開啟?",
        "answer": "phone, shower"
    }
]

```


</br>
</br>

In the example2, we add wikipedia tool enabling the LLM to access the Wikipedia API for retrieving necessary information to respond to the questions posed to it. Since the response from Wiki may contain redundant information, we can use retri_observation to retrieve relevant information.

```python

ao = akasha.test_agent(
        verbose=True,
        tools=[input_tool,
               akasha.get_saveJSON_tool(),
               akasha.get_wiki_tool()],
        retri_observation=True,
        model="openai:gpt-4")
print(ao("請用中文回答李遠哲跟黃仁勳誰比較老?將查到的資訊和答案儲存成json檔案，檔名為AGE.json"))

```

```text
根據查到的資訊，李遠哲（Yuan T. Lee）比黃仁勳（Jensen Huang）更老。李遠哲於1936年11月19日出生，而黃仁勳的出生日期是1963年2月17日。我已將這些資訊儲存成名為"AGE.json"的
JSON檔案。

### AGE.json ###
{
    "李遠哲": "1936-11-19",
    "黃仁勳": "1963-02-17",
    "答案": "李遠哲比黃仁勳更老"
}
```

</br>
</br>


# akasha_ui

If you prefer running Akasha via a web page, we offer a Streamlit-based user interface.

<br/>

To start the application, use the following command:

```bash
$ akasha ui
```

<br/>
<br/>

You should now be able to access the web page at http://localhost:8501/.

<br/>
<br/>

You can start by going to the **Settings page** to configure your settings.

<br/>

The first option, **Document Path** , specifies the directory where you want the LLM to search for documents.

<br/>

You can either add document files and name the directory from the **Upload Files** page or place the directory containing documents in the ./docs/ directory.


![image](https://github.com/iii-org/akasha/blob/master/pic/ui_setting.png)
![image](https://github.com/iii-org/akasha/blob/master/pic/ui_upload.png)


<br/>
<br/>

You can download the models you want into model/ directory, and they will be added to **Langauage Model** option in the Setting page.

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_model.png)

<br/>
<br/>

The default setting is to use the OpenAI model and embeddings, so please remember to add your OpenAI API key on the left side.

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_openai.png)


<br/>
<br/>

After you have finished setting up, you can start using Akasha.

<br/>

For example, you can instruct the language model with a query like '五軸是什麼,' and you can include a system prompt to specify how you want the model to answer in Chinese.

<br/>

It's important to note that the difference between a prompt and a system prompt is that the system prompt is not used for searching similar documents; it's more about defining the format or type of response you expect from the language model for a given prompt question.


<br/>
<br/>


![image](https://github.com/iii-org/akasha/blob/master/pic/ui_5.png)
