# Akasha

![Akasha Logo](https://github.com/iii-org/akasha/blob/master/pic/akasha_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/akasha-terminal?style=flat)](https://pypi.org/project/akasha-terminal/)
[![Downloads](https://img.shields.io/pypi/dm/akasha-terminal?style=flat&color=brightgreen)](https://pypi.org/project/akasha-terminal/)
[![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue?style=flat)](https://www.python.org/downloads/release/python-380/)
[![GitLab CI](https://img.shields.io/badge/GitLab%20CI-passing-orange?style=flat&logo=gitlab&logoColor=white)](https://gitlab.com)

<br/>
<br/>

# 快速導覽

- [介紹](#介紹)
- [更新](#更新)
  - [版本 0.8.63](#版本-0863)
  - [版本 0.8.56](#版本-0856)
  - [版本 0.8.53](#版本-0853)
- [安裝](#安裝)
- [API 金鑰](#api-金鑰)
  - [OpenAI](#openai)
  - [Azure OpenAI](#azure-openai)
- [使用範例](#使用範例)
- [可用模型](#可用模型)
- [檔案摘要](#檔案摘要)
- [代理人使用](#代理人使用)
- [Akasha UI](#akasha-ui)

<br/>
<br/>

# 介紹

Akasha 提供兩個高階功能：

1. 基於文件的問答（QA）和檢索增強生成（RAG），使用先進的大型語言模型（LLMs）。
2. 彈性選擇多種語言模型、嵌入模型和搜尋方法，以達到最佳結果。

你可以利用流行的語言模型和嵌入模型來擴展 Akasha 的功能，使其適應你的特定需求。

更多資訊及入門指南請參考我們的中文手冊 [這裡](https://tea9297.github.io/)。

<br/>
<br/>

# 更新

- 0.8.63
  - 在 Doc_QA 中新增 `ask_image` 函數，可以使用圖片作為提示進行提問。
  - 新增 `max_output_tokens` 參數。
  - 遠端相容 vllm 伺服器。

- 0.8.56
  - 新增 `extract_db_by_file` 和 `extract_db_by_keyword` 函數。

- 0.8.53
  1. Akasha Doc_QA 新增 `stream` 參數，如果 `stream=True`，返回值為生成器。
  2. `get_response` 和 `ask_self` 新增 `history_messages` 參數，用於傳遞聊天記錄給模型。
  3. `prompt_format_type` 新增 `chat_gpt` 和 `chat_mistral`，可以將提示、系統提示和聊天記錄轉換為指令格式：`[{"role":current_role, "content":prompt}]`
  4. 新增輔助函數 `call_model`、`call_batch_model`、`call_stream_model`
  5. 允許將模型物件（LLM）和嵌入物件（Embeddings）傳遞給 Doc_QA、Eval 和 Summary，避免重複定義物件。
  6. 新增輔助函數 `self_rag`
  7. Doc_QA 的 `ask_self` 和 `ask_whole_file` 可以允許文件（info）超過 `max_doc_len`，透過多次調用模型進行處理。
  8. Doc_QA 的 `get_response` 可以傳遞 `dbs` 物件給 `doc_path`，避免重新加載 ChromaDB 資料。
  9. 在 Akasha 中新增 FastAPI，你可以使用命令 "akasha api (--port port --host host --workers num_of_workers)" 啟動 FastAPI。

<br/>
<br/>

# 安裝

建議使用 Python 3.9 來運行 Akasha。你可以使用 Anaconda 創建虛擬環境。

```bash
# 建立環境
conda create --name py3-9 python=3.9
activate py3-9

# 安裝 Akasha
pip install akasha-terminal
```

<br/>
<br/>

## API 金鑰

### OPENAI

如果你想使用 OpenAI 的模型或嵌入，請前往 [OpenAI 的 API 金鑰頁面](https://platform.openai.com/account/api-keys) 獲取 API 金鑰。

你可以將金鑰儲存為 `.env` 文件中的 `OPENAI_API_KEY=your_api_key`，或者在 Bash 中使用 `export` 設定環境變數，或者在 Python 中使用 `os.environ`。

```bash
# 設定環境變數
export OPENAI_API_KEY="your api key"
```

<br/>
<br/>

### AZURE OPENAI

如果你想使用 Azure OpenAI，請前往 [Azure OpenAI Portal](https://oai.azure.com/portal) 獲取語言 API 的基礎 URL 和金鑰。

請確保已在 [Azure OpenAI Studio](https://oai.azure.com/portal) 中設定好所有模型，部署名稱應與模型名稱一致。將以下資訊儲存在 `.env` 文件中：

- `OPENAI_API_KEY=your_azure_key`
- `OPENAI_API_BASE=your_language_api_base_url`
- `OPENAI_API_TYPE=azure`
- `OPENAI_API_VERSION=2023-05-15`

 請到[這裡](https://www.genspark.ai/spark?id=98f398f3-2f6b-44d8-bfe0-40a26da12c2e)查看申請 azure api的相關步驟

<br/>

如果你同時想儲存 OpenAI 和 Azure 的金鑰，也可以使用 `AZURE_API_KEY`、`AZURE_API_BASE`、`AZURE_API_TYPE`、`AZURE_API_VERSION`。

```sh
## .env 文件
AZURE_API_KEY={your azure key}
AZURE_API_BASE={your Language API base url}
AZURE_API_TYPE=azure
AZURE_API_VERSION=2023-05-15
```

<br/>
<br/>

現在可以在 Python 中運行 Akasha：

```python
# Python 3.9
import akasha
ak = akasha.Doc_QA(model="openai:gpt-4")
response = ak.get_response(dir_path, prompt)
```

<br/>
<br/>

# 使用範例

確保更新程式碼中的 `dir_path` 和 `prompt` 變數：

- `dir_path`：更改為包含相關 PDF 文件的資料夾。
- `prompt`：設為你想要問的具體問題。

範例如下：

```python
import akasha
import os

os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?"
ak = akasha.Doc_QA(search_type="auto")
response = ak.get_response(dir_path, prompt)
print(response)
```

```python
「塞西莉亞花」的花語為「浪子的真情」
```

<br/>
<br/>

## 可用模型

以下是一些可用的模型：

```python
# OpenAI 模型（需要環境變數 'OPENAI_API_KEY' 或 'AZURE_API_KEY'）
openai_model = "openai:gpt-3.5-turbo"  # GPT-3.5 (需要 API 金鑰)
openai4_model = "openai:gpt-4"          # GPT-4 (需要 API 金鑰)

# Hugging Face 模型（需要環境變數 'HUGGINGFACEHUB_API_TOKEN'）
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf"  # Meta Llama-2 模型（需要 API 金鑰）

# 量化模型（需要 GPU 以獲得最佳效能）
quantized_ch_llama_model = "gptq:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"  # 4 位元量化 Llama2 中文模型

# ChatGLM 模型
chatglm_model = "chatglm:THUDM/chatglm2-6b"  # THUDM ChatGLM2 模型
```

<br/>

## 可用的嵌入模型

這些是一些可用的嵌入模型：

```python
openai_emd = "openai:text-embedding-ada-002"  # 需要 "OPENAI_API_KEY"; 最大序列長度 8192
huggingface_emd = "hf:all-MiniLM-L6-v2"
text2vec_ch_emd = "hf:shibing624/text2vec-base-chinese"  # 最大序列長度 128
```

<br/>
<br/>

## 檔案摘要

若要對文本檔案（如 .pdf、.txt 或 .docx）進行摘要，可以使用 `Summary.summarize_file` 函數：

```python
import akasha
sum = akasha.Summary(chunk_size=1000, chunk_overlap=100)
sum.summarize_file(file_path="doc/mic/5軸工具機因應市場訴求改變的發展態勢.pdf", summary_type="map_reduce", summary_len=500, chunk_overlap=40)
```

<br/>
<br/>

## 代理人使用

通過實現代理人，可以使 LLM 更有效地使用工具完成任務，例如檔案編輯、Google 搜尋等。

<br/>
在示例中，我們創建了一個工具可以收集用戶輸入，並使用代理將用戶輸入存儲到 JSON 文件中：

```python
def input_func(question: str):
    response = input(question)
    return str({"question": question, "answer": response})

input_tool = akasha.create_tool(
    "user_question_tool",
    "這是向用戶提問的工具，參數 question 是尚未回答的問題字串。",
    func=input_func)

ao = akasha.test_agent(verbose=True, tools=[input_tool, akasha.get_saveJSON_tool()], model="openai:gpt-4")
print(ao("逐個詢問使用者以下問題，若所有問題都回答了，則將所有問題和回答儲存成default.json並結束。問題為:1.房間燈關了嗎? 2.有沒有人在家? 3.有哪些電器開啟?"))
```

```text
我已成功將所有問題和回答儲存到 "default.json" 文件中，對話已完成。
```

<br/>
<br/>

# Akasha UI

如果你希望透過網頁使用 Akasha，我們提供了基於 Streamlit 的用戶界面。

<br/>
使用以下命令啟動應用程式：

```bash
$ akasha ui
```

<br/>
現在可以訪問 http://localhost:8501/。

可以從 **設定頁面** 開始進行設置，如設置文件目錄、模型等，然後開始使用 Akasha。

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_setting.png)

<br/>
<br/>
你可以在 **上傳檔案頁面** 中添加文件，也可以直接將包含文件的資料夾放在 ./docs/ 目錄下。

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_upload.png)

<br/>
<br/>
將你想下載的模型放到 model/ 目錄中，這樣它們就會出現在設定頁面的 **語言模型** 選項中。

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_model.png)

<br/>
<br/>
預設設定使用 OpenAI 模型和嵌入，因此請記得在左側添加你的 OpenAI API 金鑰。

![image](https://github.com/iii-org/akasha/blob/master/pic/ui_openai.png)

<br/>
<br/>
設置完成後，你可以開始使用 Akasha。例如，可以用「五軸是什麼」進行提問，並使用系統提示來指定希望模型用中文回答。

<br/>
<br/>
![image](https://github.com/iii-org/akasha/blob/master/pic/ui_5.png)

