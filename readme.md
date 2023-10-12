# akasha

Akasha simplifies document-based Question Answering (QA) by harnessing the power of Large Language Models to accurately answer your queries while searching through your provided documents.

With Akasha, you have the flexibility to choose from a variety of language models, embedding models, and search types. Adjusting these parameters is straightforward, allowing you to optimize your approach and discover the most effective methods for obtaining accurate answers from Large Language Models.


<br/>
<br/>

<br/>
<br/>

# Installation

We recommend using Python 3.8 to run our akasha package. You can use Anaconda to create virtual environment.

```bash

# create environment

conda create --name py3-8 python=3.8
activate py3-8

# install akasha
pip install git+https://gitlab-devops.iii.org.tw/root/qaiii-1.git


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

```python
#PYTHON3.8
import akasha
ak = akasha.Doc_QA(model="openai:gpt-3.5-turbo")
response = ak.get_response(dir_path, prompt)


```


<br/>
<br/>


### LLAMA-2 
If you want to use original meta-llama model, you need to both register to [huggingface](https://huggingface.co/settings/tokens) to get access token and [meta-llama](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to request access. 

**Remember, the account on Hugging Face and the email you use to request access to Meta-Llama must be the same, so that you can download models from Hugging Face once your account is approved.**

You should see the **Gated model You have been granted access to this model** once your account is approved
![image](pic/granted.png)



<br/>
<br/>

Again, you can either save **HUGGINGFACEHUB_API_TOKEN=your api key** into **.env** file to current working directory or set as a environment variable, using **export** in bash or use **os.environ** in python. 
After you create Doc_QA() class, you can still change the model you want when you call the function.

```bash

# set a environment variable

export HUGGINGFACEHUB_API_TOKEN="your api key"


```

```python
#PYTHON3.8
import akasha
ak = akasha.Doc_QA()
response = ak.get_response(dir_path, prompt, model="hf:meta-llama/Llama-2-7b-chat-hf")


```


<br/>
<br/>







# Example and Parameters

## Basic get_response OpenAI example


``` python
import akasha
import os


os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?	「失之交臂的感情」	「赤誠的心」	「浪子的真情」	「無法挽回的愛」"
ak = akasha.Doc_QA()
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


## Select different embeddings

Using parameter "embeddings", you can choose different embedding models, and the embedding model will be used to store documents into vector storage and search relevant documents from prompt.  Default is **openai:text-embedding-ada-002**.

Currently support **openai**, **huggingface** and **tensorflowhub**.

### huggingface example

``` 
ak = akasha.Doc_QA(embeddings="huggingface:all-MiniLM-L6-v2")
resposne = ak.get_response(dir_path, prompt)
```
To use huggingface embedding models, you can type huggingface:model_name or hf:model_name, for example, **huggingface:all-MiniLM-L6-v2**

<br/>
<br/>
<br/>
<br/>


## Select different models
Using parameter **"model"**, you can choose different text generation models, default is **openai:gpt-3.5-turbo**.

Currently support **openai**, **llama-cpp** and **huggingface**.

### 1.huggingface example
``` python
ak = akasha.Doc_QA()
ak.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="hf:meta-llama/Llama-2-13b-chat-hf")

```

To use text generation model from **huggingface**, for example, meta llama, you can type **hf:meta-llama/Llama-2-13b-chat-hf**


<br/>
<br/>

### 2.llama-cpp example
llama-cpp can use quantized llama model and run on cpu, after you download or transfer llama-cpp model file using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

```python
ak = akasha.Doc_QA()
ak.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="llama-cpu:model/llama-2-13b-chat.Q5_K_S.gguf")

```

For example, if q5 model is in the "model/" directory, you can assign **llama-cpu:model/llama-2-13b-chat.Q5_K_S.gguf** to load model.

```python
ak = akasha.Doc_QA()
ak.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="llama-gpu:model/llama-2-3b-chat.Q5_K_S.gguf")

```
you can also combine gpu with cpu to run llama-cpp, using **llama-gpu:model/llama-2-13b-chat.Q5_K_S.gguf**

<br/>
<br/>
<br/>
<br/>


## Select different search type
Using parameter **"search_type"**, you can choose different search methods to find similar documents , default is **merge**, which is
the combination of mmr, svm and tfidf. Currently you can select merge, mmr, svm and tfidf.

**Max Marginal Relevance(mmr)** select similar documents by cosine similarity, but it also consider diversity, so it will also penalize document for closeness to already selected documents.

**Support Vector Machines(svm)** use the input prompt and the documents vectors to train svm model, after training, the svm can be used to score new vectors based on their similarity to the training data. 

**Term Frequency–Inverse Document Frequency(tfidf)** is a commonly used weighting technique in information retrieval and text mining. TF-IDF is a statistical method used to evaluate the importance of a term in a collection of documents or a corpus with respect to one specific document in the collection.


``` python
ak = akasha.Doc_QA(search_type="merge")
akasha.get_response(dir_path, prompt, search_type="mmr")

```
 
<br/>
<br/>
<br/>
<br/>



## Some models you can use

Please note that for OpenAI models, you need to set the environment variable 'OPENAI_API_KEY,' and for most Hugging Face models, a GPU is required to run the models. However, for .gguf models, you can use a CPU to run them.

```python
openai_model = "openai:gpt-3.5-turbo"  # need environment variable "OPENAI_API_KEY"
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf" #need environment variable "HUGGINGFACEHUB_API_TOKEN" to download meta-llama model
quantized_ch_llama_model = "hf:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"
taiwan_llama_gptq = "hf:weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"
### If you want to use llama-cpp to run model on cpu, you can download gguf version of models 
### from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF  and the name behind "llama-gpu:" or "llama-cpu:"
### from https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF
### is the path of the downloaded .gguf file
llama_cpp_model = "llama-gpu:model/llama-2-13b-chat-hf.Q5_K_S.gguf"  
llama_cpp_model = "llama-cpu:model/llama-2-7b-chat.Q5_K_S.gguf"
llama_cpp_chinese_alpaca = "llama-gpu:model/chinese-alpaca-2-7b.Q5_K_S.gguf"
llama_cpp_chinese_alpaca = "llama-cpu:model/chinese-alpaca-2-13b.Q5_K_M.gguf"
chatglm_model = "chatglm:THUDM/chatglm2-6b"


```



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




# Functions

## Use chain-of-thought to solve complicated problem
 
instead of input one single prompt, you can input multiple small stop questions to get better answer.  
 
 
```python
import akasha
import os


os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "mic/"
queries2 = ["西門子自有工廠如何朝工業4.0 發展","詳細解釋「工業4.0 成熟度指數」發展路徑的六個成熟度","根據西門子自有工廠朝工業4.0發展，探討其各項工業4.0的成熟度指標"]
ak = akasha.Doc_QA()
response = ak.chain_of_thought(dir_path, queries2, search_type='svm')
print(response)
	
	
```


```text
response 1:
西門子自有工廠朝工業4.0發展的方式包括以下幾個方面：

1. 數位化戰略：西門子提出數位化戰略，從工業4.0策略擬定到落地執行，為客戶提供一條龍服務。他們設計數位工廠原型
，搭配OT、IT方案，並使用西門子的MindSphere工業物聯網平台，發展數據可視化和數據分析相關應用。

2. 跨領域合作：西門子近年積極與雲服務商、系統商等跨領域合作，推動智慧製造解決方案。此外，他們也與SAP進行ERP整合，專注於物聯網領域。

3. 虛實整合：西門子在中國大陸成都生產研發基地的案例中，從研發、生產、訂單管理、供應商管理到物流作業
，實現了整條價值鏈的虛實整合。他們不斷提高配料、傳輸、檢測等流程的自動化程度。

總體而言，西門子通過數位化戰略、跨領域合作和虛實整合等方式，推動自有工廠朝工業4.0發 
展。他們致力於提升生產效率和效能，並利用先進的技術和解決方案實現智慧工廠的建設。






response 2:
「工業4.0成熟度指數」的發展路徑劃分為六個成熟度，分別是電腦化、可連結、可視化、可分析、可預測和自適應。

1. 電腦化：這是工業4.0發展的起點，指企業開始使用計算機技
術，人員不再手動操作機械。然而，機械仍未聯網，各個IT系統仍各自獨立，資料尚未串聯。例如，企業的ERP系統與生產相關的系統獨立運作，訂單與產品品檢紀錄分散於兩套系統，導致 
訂單無法回溯出現品質問題的環節。

2. 可連結：在這個成熟度階段，企業開始將各個IT系統進行連接，實現資料的串聯。這使得不同系統之間可以共享資料，提高資訊的流通效率。例 
如，企業的ERP系統與生產相關的系統進行連接，訂單與產品品檢紀錄可以實現資料的回溯。

3. 可視化：在這個成熟度階段，企業開始實現資料的可視化，將資料以圖形化或圖表化的方
式呈現，使得管理者可以直觀地了解企業的運營狀況。例如，企業可以使用數據儀表板或報表來呈現生產線的運行情況和產品的品質指標。

4. 可分析：在這個成熟度階段，企業開始進 
行資料的分析，利用數據分析工具和算法來挖掘資料中的價值和洞察。這使得企業可以更深入地了解生產過程中的問題和潛在的改進空間。例如，企業可以使用數據分析工具來分析生產線的
效率和品質問題，並提出改進措施。

5. 可預測：在這個成熟度階段，企業開始利用資料分析的結果來進行預測和預測模型的建立。這使得企業可以預測生產過程中可能出現的問題，並 
提前採取相應的措施。例如，企業可以利用預測模型來預測生產線的故障和產品的品質問題，並提前進行維護和調整。

6. 自適應：在這個成熟度階段，企業開始實現自動化和自適應能 
力，使得生產過程可以根據實時的數據和環境變化進行調整和優化。這使得企業可以更靈活地應對市場需求和生產變化。例如，企業可以實現生產線的自動調整和產品的自動優化，以適應市
場需求的變化。

這六個成熟度階段代表了企業在工業4.0發展過程中的不同階段和能力水平，企業可以根據自身的情況和目標，逐步提升成熟度，實現工業4.0的目標。






response 3:
根據西門子自有工廠朝工業4.0發展的方式，可以探討其在工業4.0成熟度指標中的幾個方面：

 

1. 數位化戰略：西門子提出數位化戰略，從工業4.0策略擬定到落地執行提供一條龍服務
。這代表企業在工業4.0成熟度指標中已經達到了可連結和可視化的階段，並開始將數據應用於生產優化和資源利用。

 

2. 整合系統：西門子在廠內進行軟體間整合，包括PLM、ERP、MOM 
、WMS和Automation五大系統的整合，使數據互聯互通。這代表企業在工業4.0成熟度指標中已經達到了可分析和可預測的階段，並能夠利用數據分析技術進行深入分析和預測。

 

3. 數據 應用：西門子利用自有的數位雙生軟體Tecnomatix，打造虛擬工廠，模擬生產狀況或監控實際生產狀況。這代表企業在工業4.0成熟度指標中已經達到了可分析和可預測的階段，並能夠利用 
數據應用提供的資訊，優化生產設備和工序。

 

總的來說，根據西門子自有工廠朝工業4.0發展的方式，可以看出他們在工業4.0成熟度指標中已經達到了可連結、可視化、可分析和可預測，優化生產設備和工序。
```

<br/>
<br/>


```python
### Arguments of Doc_QA class ###
"""
Args:
            **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
                includes 'merge', 'mmr', 'svm', 'tfidf', also, you can custom your own search_type function, as long as your
                function input is (query_embeds:np.array, docs_embeds:list[np.array], k:int, relevancy_threshold:float, log:dict) 
                and output is a list [index of selected documents].\n
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set record_exp as experiment name.  default "".\n
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".\n
            **max_token (int, optional)**: max token size of llm input. Defaults to 3000.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n

"""
```




<br/>
<br/>


<br/>
<br/>


## Save Logs
Each time you run any function from akasha, it will save logs that record the parameters of this run and the results. Each run will have a timestamp, you can use {obj_name}.timestamp_list to check them, and use it to find the log of the run you want see.

You can also save logs into .txt file or .json file

```python
qa = akasha.Doc_QA(verbose=False, search_type="merge", max_token=2500,model="llama-gpu:model/chinese-alpaca-2-7b.Q5_K_S.gguf")
query1 = "五軸是什麼"
qa.get_response(doc_path="./doc/mic/", prompt = query1)
qa.get_response(doc_path="./doc/mic/", prompt = query1)

tp = qa.timestamp_list
print(tp)
## ["2023/09/26, 10:52:36", "2023/09/26, 10:59:49", "2023/09/26, 11:09:23"]

print(qa.logs[tp[-1]])
## {"fn_type":"get_response","search_type":"merge", "max_token":2500,....."response":....}


qa.save_logs(file_name="logs.json",file_type="json")
```


![image](pic/logs.png)







<br/>
<br/>


<br/>
<br/>

## Use AiiDO to record experiment 


If you want to record experiment metrics and results, you need to create a project on the AiiDO platform. Once done, 
you will receive all the necessary parameters for automatically uploading the experiment. 

Create a .env file on the same directory of your program, and paste all parameters.


.env file

```python
MINIO_URL= YOUR_MINIO_URL
MINIO_USER= YOUR_MINIO_USER
MINIO_PASSWORD= YOUR_MINIO_PASSWORD
TRACKING_SERVER_URI= YOUR_TRACKING_SERVER_URI
```
<br/>
<br/>

After you created .env file, you can use **record_exp** to set your experiment name and it will automatically record 
experiment metrics and results to mlflow server.

```python
import akasha
import os
from dotenv import load_dotenv
load_dotenv() 

os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?	「失之交臂的感情」	「赤誠的心」	「浪子的真情」	「無法挽回的愛」"
exp_name = "exp_akasha_get_response"
ak = akasha.Doc_QA(record_exp=exp_name)
response = ak.get_response(dir_path, prompt)

```

<br/>
<br/>


<br/>
<br/>



### In an experiment you assign, the run name is the combinations of the usage of embedding, search type and model name

![image](pic/upload_experiments.png)


<br/>
<br/>
<br/>
<br/>

### You can also compare the responses from different models, search type and embeddings


![image](pic/response_comparison.png)


<br/>
<br/>
<br/>
<br/>






## Auto Evaluation

To evaluate the performance of current parameters, you can use function **auto_evaluation** . First you need to build a question set .txt
file based on the documents you want to use.
You can either generate **single choice question file** or **essay question file**.
  1. For **single choice question file**, every options and the correct answer is separated by tab(\t), each line is a question, 
  **for example:**  (question_pvc.txt)

  ```text

  應回收廢塑膠容器材質種類不包含哪種?	聚丙烯（PP）	聚苯乙烯（PS）	聚氯乙烯（PVC）	低密度聚乙烯（LDPE）	4
  庫存盤點包括庫存全盤作業及不定期抽盤作業，盤點計畫應包括下列項目不包含哪項?	盤點差異之處理	盤點清冊	各項物品存放區域配置圖	庫存全盤日期及參加盤點人員名單	1
  以下何者不是環保署指定之公民營地磅機構?	中森加油站企業有限公司	台益地磅站	大眾地磅站	新福行	4

  ```

  it will return the correct rate and tokens of the question set, details of each question would save in logs, or in mlflow server if you
  turn on **record_exp** 

  ```python
  import akasha.eval as eval
  import os
  from dotenv import load_dotenv
  load_dotenv() 

  os.environ["OPENAI_API_KEY"] = "your openAI key"
  dir_path = "doc/pvc/"
  exp_name = "exp_akasha_auto_evaluation"

  eva = eval.Model_Eval(question_type="single_choice", search_type='merge',\
      model="openai:gpt-3.5-turbo", embeddings="openai:text-embedding-ada-002",record_exp=exp_name)
  print(eva.auto_evaluation("question_pvc.txt", dir_path ))
  ## correct rate: 0.9, tokens: 3228 ##
  ```

  <br/>
  <br/>
  
  2. For **essay question file** , each question has "問題：" before it, and each reference answer has "答案：" before it. 
  Each question is separated by two newline(\n\n)

  ```text

  問題：根據文件中的訊息，智慧製造的複雜性已超越系統整合商的負荷程度，未來產業鏈中的角色將傾向朝共和共榮共創智慧製造商機，而非過往的單打獨鬥模式發展。請問為什麼  供  應商、電信商、軟體開發商、平台商、雲端服務供應商、系統整合商等角色會傾向朝共和共榮共創智慧製造商機的方向發展？
  答案：因為智慧製造的複雜性已超越系統整合商的負荷程度，單一角色難以完成整個智慧製造的需求，而共和共榮共創的模式可以整合各方的優勢，共同創造智慧製造的商機。

  問題：根據文件中提到的資訊技術商（IT）和營運技術商（OT），請列舉至少兩個邊緣運算產品或解決方案。
  答案：根據文件中的資訊，NVIDIA的邊緣運算產品包括Jetson系列和EGX系列，而IBM的邊緣運算產品包括IBM Edge Application Manager和IBM Watson Anywhere。
  ```


<br/>
<br/>
<br/>
<br/>






## Use llm to create questionset and evaluate the performance
If you prefer not to create your own question set to assess the performance of the current parameters, you can utilize the **eval.auto_create_questionset** feature to automatically generate a question set along with reference answers. Subsequently, you can use **eval.auto_evaluation** to obtain metrics scores such as **Bert_score**, **Rouge**, and **LLM_score** for essay questionset and **correct rate** for single choice questionset. These scores range from 0 to 1, with higher values indicating that the generated response closely matches the reference answers.

For example, the code create a questionset text file 'mic_1.txt' with ten questions and reference answers, each question is randomly generated from the content segments of given documents in 'doc/mic/' directory. Then you can use the questionset text file to evaluate the performance of the parameters you want to test.


```python

import akasha.eval as eval

eva = eval.Model_Eval(question_type="essay", search_type='merge',\
      model="openai:gpt-3.5-turbo", embeddings="openai:text-embedding-ada-002",record_exp="exp_mic_auto_questionset")

eva.auto_create_questionset(doc_path="doc/mic/", question_num=10)

bert_score, rouge, llm_score = eva.auto_evaluation(questionset_path="questionset/mic_1.txt", doc_path="doc/mic/", question_type = "essay", record_exp="exp_mic_auto_evaluation",topK=3,search_type="svm")

# bert_score = 0.782
# rouge = 0.81
# llm_score = 0.393
```


<br/>
<br/>
<br/>
<br/>




## Find Optimum Combination

To test all available combinations and find the best parameters, you can use function **optimum_combination** , you can give different 
embeddings, document chunk sizes, models, document similarity searching type and number of most relative documents (topK), and the function will
test all combinations to find the best combination based on the given question set and documents. 

Noted that best score combination is the highest correct rate combination, and best cost-effective 
combination is the combination that need least tokens to get a correct answer.


```python
import akasha.eval as eval
import os
from dotenv import load_dotenv
load_dotenv() 

os.environ["OPENAI_API_KEY"] = "your openAI key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your huggingface key"
dir_path = "doc/pvc/"
exp_name = "exp_akasha_optimum_combination"
embeddings_list = ["hf:shibing624/text2vec-base-chinese", "openai:text-embedding-ada-002"]
model_list = ["openai:gpt-3.5-turbo","hf:FlagAlpha/Llama2-Chinese-13b-Chat-4bit","hf:meta-llama/Llama-2-7b-chat-hf",\
            "llama-gpu:model/llama-2-7b-chat.Q5_K_S.gguf", "llama-gpu:model/llama-2-13b-chat.Q5_K_S.gguf"]

eva = eval.Model_Eval(question_type="single_choice")
eva.optimum_combination("question_pvc.txt", dir_path,  embeddings_list = embeddings_list, model_list = model_list,
            chunk_size_list=[200, 400, 600], search_type_list=["merge","tfidf",],record_exp=exp_name,topK_list=[2,3])

```

**The result would look like below**

```text

Best correct rate:  1.000
Best score combination:  

embeddings: openai:text-embedding-ada-002, chunk size: 400, model: openai:gpt-3.5-turbo, topK: 3, search type: merge

 

embeddings: openai:text-embedding-ada-002, chunk size: 400, model: openai:gpt-3.5-turbo, topK: 3, search type: tfidf

 

 

Best cost-effective:

embeddings: hf:shibing624/text2vec-base-chinese, chunk size: 400, model: openai:gpt-3.5-turbo, topK: 2, search type: tfidf

```

<br/>
<br/>

```python
"""
### Arguments of Model_Eval class ###
 Args:
            **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".
            **verbose (bool, optional)**: show log texts or not. Defaults to False.
            **topK (int, optional)**: search top k number of similar documents. Defaults to 2.
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.
            **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
                includes 'merge', 'mmr', 'svm', 'tfidf', also, you can custom your own search_type function, as long as your
                function input is (query_embeds:np.array, docs_embeds:list[np.array], k:int, relevancy_threshold:float, log:dict) 
                and output is a list [index of selected documents].
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set record_exp as experiment name.  default "".
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".
            **max_token (int, optional)**: max token size of llm input. Defaults to 3000.
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".
"""
```




<br/>
<br/>
<br/>
<br/>





## File Summarization
To create a summary of a text file in various formats like .pdf, .txt, or .docx, you can use the **summary.summarize_file** function. For example, the following code employs the **map_reduce** summary method to instruct LLM to generate a summary of approximately 500 words.

There're two summary type, **map_reduce** and **refine**, **map_reduce** will summarize every text chunks and then use all summarized text chunks to generate a final summary; **refine** will summarize each text chunk at a time and using the previous summary as a prompt for 
summarizing the next segment to get a higher level of summary consistency.

```python

import akasha.summary as summary
sum = summary.Summary( chunk_size=1000, chunk_overlap=100)
sum.summarize_file(file_name="doc/mic/5軸工具機因應市場訴求改變的發展態勢.pdf",summary_type="map_reduce", summary_len=500\
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
            **max_token (int, optional)**: max token size of llm input. Defaults to 3000.
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.
"""
```

<br/>
<br/>
<br/>
<br/>

# Command Line Interface
You can also use akasha in command line, for example, you can use **keep-responsing** to create a document QA model 
and keep asking different questions and get response based on the documents in the given -d directory.



```console
$ akasha keep-responsing -d ../doc/plc/  -c 400 -k 1
Please input your question(type "exit()" to quit) : 應回收廢塑膠容器材質種類不包含哪種?  聚丙烯（PP） 聚苯乙烯（PS） 聚氯乙烯（PVC）  低密度聚乙烯（LDPE）
Response:  應回收廢塑膠容器材質種類不包含低密度聚乙烯（LDPE）。



Please input your question(type "exit()" to quit) : 所謂市盈率，是指每股市價除以每股盈餘，也就是股票的?   本益比  帳面值比  派息   資金
英國和德國等多個市場。然而，義大利、加拿大和澳洲並不在這些可交易的國家之列。



Please input your question(type "exit()" to quit) : exit()

```

<br/>
<br/>

Currently you can use **get-response**, **keep-responsing**, **chain-of-thought** and **auto_create_questionset** and **auto_evaluation**.
  


```bash
$ akasha keep-responsing --help
Usage: akasha keep-responsing [OPTIONS]

Options:
  -d, --doc_path TEXT         document directory path, parse all .txt, .pdf,
                              .docx files in the directory  [required]
  -e, --embeddings TEXT       embeddings for storing the documents
  -c, --chunk_size INTEGER    chunk size for storing the documents
  -m, --model TEXT            llm model for generating the response
  -k, --topk INTEGER          select topK relevant documents
  -t, --threshold FLOAT       threshold score for selecting the relevant
                              documents
  -l, --language TEXT         language for the documents, default is 'ch' for
                              chinese
  -s, --search_type TEXT      search type for the documents, include merge,
                              svm, mmr, tfidf
  -sys, --system_prompt TEXT  system prompt for the llm model
  -mt, --max_token INTEGER    max token for the llm model input
  --help                      Show this message and exit.

```

<br/>
<br/>
<br/>
<br/>


# akasha_ui

If you prefer running Akasha via a web page, we offer a Streamlit-based user interface.

<br/>

To get started, you can download the project and install it in the same directory as the setup.py script.


```bash

$ git clone https://gitlab-devops.iii.org.tw/root/qaiii-1.git
$ cd qaiii-1/
$ python -m pip install .

```

<br/>
<br/>



Next, install Streamlit and Streamlit-option-menu. Then, use Streamlit to run the ui.py script.

<br/>

You should now be able to access the web page at http://localhost:8501/.

<br/>
<br/>


```bash

$ python -m pip install streamlit
$ python -m pip install streamlit-option-menu
$ streamlit run ui.py

```

<br/>
<br/>

You can start by going to the **Settings page** to configure your settings. 

<br/>

The first option, **Document Path** , specifies the directory where you want the LLM to search for documents.

<br/>

You can either add document files and name the directory from the **Upload Files** page or place the directory containing documents in the ./docs/ directory.


![image](pic/ui_setting.png)
![image](pic/ui_upload.png)


<br/>
<br/>

You can download the models you want into model/ directory, and they will be added to **Langauage Model** option in the Setting page.

![image](pic/ui_model.png)

<br/>
<br/>

The default setting is to use the OpenAI model and embeddings, so please remember to add your OpenAI API key on the left side.

![image](pic/ui_openai.png)


<br/>
<br/>

After you have finished setting up, you can start using Akasha. 

<br/>

For example, you can instruct the language model with a query like '五軸是什麼,' and you can include a system prompt to specify how you want the model to answer in Chinese. 

<br/>

It's important to note that the difference between a prompt and a system prompt is that the system prompt is not used for searching similar documents; it's more about defining the format or type of response you expect from the language model for a given prompt question.


<br/>
<br/>


![image](pic/ui_5.png)
