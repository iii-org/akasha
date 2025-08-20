# akasha

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi package](https://img.shields.io/pypi/v/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![downloads](https://img.shields.io/pypi/dm/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![python version : 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-3109/)
![GitLab CI](https://img.shields.io/badge/gitlab%20ci-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white)

<br/>

akasha simplifies document-based Question Answering (QA) and Retrieval Augmented Generation(RAG) by harnessing the power of Large Language Models to accurately answer your queries while searching through your provided documents.

With akasha, you have the flexibility to choose from a variety of language models, embedding models, and search types. Adjusting these parameters is straightforward, allowing you to optimize your approach and discover the most effective methods for obtaining accurate answers from Large Language Models.

For the chinese manual, please visit [manual](https://tea9297.github.io/akasha/)

<br/>
<br/>

<br/>
<br/>

# Updates

- 0.9.0

    1. new function calling
   
<br/>
<br/>

<br/>
<br/>



# Installation

We recommend using Python 3.10 to run our akasha package. You can use uv or Anaconda to create virtual environment.

```shell!
###create environment
$ uv venv --python 3.10

###install akasha
$ uv pip install akasha-terminal
```

<br/>
<br/>

## API Keys

### OPENAI

If you want to use openai models or embeddings, go to [openai](https://platform.openai.com/account/api-keys) to get the API key.
You can either save **OPENAI_API_KEY=your api key** into **.env** file to current working directory or,
set as a environment variable, using **export** in bash or use **os.environ** in python.

```bash

# set a environment variable

export OPENAI_API_KEY="your api key"


```

<br/>
<br/>

### AZURE OPENAI

If you want to use azure openai, go to [auzreAI](https://oai.azure.com/portal) and get you own Language API base url and key.
Also, remember to depoly all the models in [Azure OpenAI Studio](https://oai.azure.com/portal), the deployment name should be same as the model name. save **OPENAI_API_KEY=your azure key**,  **OPENAI_API_BASE=your Language API base url**, **OPENAI_API_TYPE=azure**, **OPENAI_API_VERSION=2023-05-15** into **.env** file to current working directory.

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
#PYTHON3.10+
import akasha
data_source = "doc/mic"
prompt = "五軸是什麼?"
ak = akasha.RAG(model="openai:gpt-4o")
response = ak(data_source, prompt)


```

<br/>
<br/>


## Some models you can use

Please note that for OpenAI models, you need to set the environment variable 'OPENAI_API_KEY,' and for most Hugging Face models, a GPU is required to run the models. However, for .gguf models, you can use a CPU to run them.

```python
openai_model = "openai:gpt-3.5-turbo"  # need environment variable "OPENAI_API_KEY" or "AZURE_API_KEY"
openai4_model = "openai:gpt-4"  # need environment variable "OPENAI_API_KEY" or "AZURE_API_KEY"
openai4o_model = "openai:gpt-4o" # need environment variable "OPENAI_API_KEY"
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf" #need environment variable "HUGGINGFACEHUB_API_TOKEN" to download meta-llama model
quantized_ch_llama_model = "gptq:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"
taiwan_llama_gptq = "gptq:weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"
mistral = "hf:Mistral-7B-Instruct-v0.2" 
mediatek_Breeze = "hf:MediaTek-Research/Breeze-7B-Instruct-64k-v0.1"
### If you want to use llama-cpp to run model on cpu, you can download gguf version of models 
### from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF  and the name behind "llama-gpu:" or "llama-cpu:"
### from https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF
### is the path of the downloaded .gguf file
llama_cpp_model = "llama-cpp:model/llama-2-13b-chat-hf.Q5_K_S.gguf"  
llama_cpp_model = "llama-cpp:model/llama-2-7b-chat.Q5_K_S.gguf"
llama_cpp_chinese_alpaca = "llama-cpp:model/chinese-alpaca-2-7b.Q5_K_S.gguf"
chatglm_model = "chatglm:THUDM/chatglm2-6b"

```

<br/>
<br/>


## Some embeddings you can use

Please noted that each embedding model has different window size, texts that over the max seq length will be truncated and won't be represent
in embedding model.

Rerank_base and rerank_large are not embedding models; instead, they compare the query to each chunk of the documents and return scores that represent the similarity. As a result, they offer higher accuracy compared to embedding models but may be slower.

```python
openai_emd = "openai:text-embedding-ada-002"  # need environment variable "OPENAI_API_KEY"  # 8192 max seq length
huggingface_emd = "hf:all-MiniLM-L6-v2" 
text2vec_ch_emd = "hf:shibing624/text2vec-base-chinese"   # 128 max seq length 
text2vec_mul_emd = "hf:shibing624/text2vec-base-multilingual"  # 256 max seq length
text2vec_ch_para_emd = "hf:shibing624/text2vec-base-chinese-paraphrase" # perform better for long text, 256 max seq length
bge_en_emd = "hf:BAAI/bge-base-en-v1.5"  # 512 max seq length
bge_ch_emd = "hf:BAAI/bge-base-zh-v1.5"  # 512 max seq length

rerank_base = "rerank:BAAI/bge-reranker-base"    # 512 max seq length
rerank_large = "rerank:BAAI/bge-reranker-large"  # 512 max seq length

```

<br/>
<br/>
<br/>
<br/>






## File Summarization

To create a summary of a text file in various formats like .pdf, .txt, or .docx, you can use the **Summary** class. For example, the following code employs the **map_reduce** summary method to instruct LLM to generate a summary of approximately 500 words.

There're two summary type, **map_reduce** and **refine**, **map_reduce** will summarize every text chunks and then use all summarized text chunks to generate a final summary; **refine** will summarize each text chunk at a time and using the previous summary as a prompt for summarizing the next segment to get a higher level of summary consistency.

```python
import akasha

summ = akasha.summary(
    model="openai:gpt-4o",
    sum_type="map_reduce",
    chunk_size=1000,
    sum_len=1000,
    language="en",
    keep_logs=True,
    verbose=True,
    max_input_tokens=8000,
)

# 總結內容，來源可以是網址、文件或純文字
ret = summ(content=["https://github.com/iii-org/akasha"])
```

<br/>
<br/>


## agent

By implementing an agent, you empower the LLM with the capability to utilize tools more effectively to accomplish tasks. You can allocate tools for tasks such as file editing, conducting Google searches, and enlisting the LLM's assistance in task execution, rather than solely relying on it to respond your questions.


<br/>
<br/>

### 使用內建工具

```python
import akasha.agent.agent_tools as at

# 使用內建的網頁搜尋工具和 JSON 保存工具
tool_list = [at.websearch_tool(search_engine="brave"), at.saveJSON_tool()]

agent = akasha.agents(
    tools=tool_list,
    model="openai:gpt-4o",
    temperature=1.0,
    max_input_tokens=8000,
    verbose=True,
    keep_logs=True,
)

# 問問題並使用工具回答
response = agent("用網頁搜尋工業4.0，並將資訊存成json檔iii.json")
print(response)

# 保存日誌
agent.save_logs("logs.json")
```



</br>
</br>

### 定義自訂工具並使用

```python
import akasha
from datetime import datetime

# 定義一個工具來獲取今天的日期
def today_f():
    now = datetime.now()
    return "today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))

# 創建工具
today_tool = akasha.create_tool(
    "This is the tool to get today's date, the tool doesn't have any input parameter.",
    today_f,
    "today_date_tool",
)

# 創建 agent 並使用工具
agent = akasha.agents(
    tools=[today_tool],
    model="openai:gpt-4o",
    temperature=1.0,
    verbose=True,
    keep_logs=True,
)

# 問問題並使用工具回答
response = agent("今天幾月幾號?")
print(response)

# 保存日誌
agent.save_logs("logs.json")
```

</br>
</br>


### 使用 MCP 伺服器上的工具


```python
import asyncio
import akasha
from langchain_mcp_adapters.client import MultiServerMCPClient

MODEL = "openai:gpt-4o"

# 定義 MCP 伺服器連接資訊
connection_info = {
    "math": {
        "command": "python",
        "args": ["cal_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/sse",
        "transport": "sse",
    },
}
prompt = "tell me the weather in Taipei"

# 使用 MCP 工具
agent = akasha.agents(
    model=MODEL,
    temperature=1.0,
    verbose=True,
    keep_logs=True,
)
response = agent.mcp_agent(connection_info, prompt)
agent.save_logs("logs_agent.json")

```
