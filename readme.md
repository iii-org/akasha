## OpenAI example


``` python
import akasha
import os


os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?	「失之交臂的感情」	「赤誠的心」	「浪子的真情」	「無法挽回的愛」"

response = akasha.get_response(dir_path, prompt)
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
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2")
```
To use huggingface embedding models, you can type huggingface:model_name or hf:model_name, for example, **huggingface:all-MiniLM-L6-v2**

<br/>
<br/>
<br/>
<br/>


## Select different models
Using parameter "model", you can choose different text generation models, default is **openai:gpt-3.5-turbo**.

Currently support **openai**, **llama-cpp** and **huggingface**.

### 1.huggingface example
``` python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="hf:meta-llama/Llama-2-13b-chat-hf")

```

To use text generation model from **huggingface**, for example, meta llama, you can type **hf:meta-llama/Llama-2-13b-chat-hf**


<br/>
<br/>

### 2.llama-cpp example
llama-cpp can use quantized llama model and run on cpu, after you download or transfer llama-cpp model file using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

```python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="llama-cpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin")

```

For example, if q4 model is in the "model/" directory, you can assign **llama-cpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin** to load model.

```python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="llama-gpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin")

```
you can also combine gpu with cpu to run llama-cpp, using **llama-gpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin**

<br/>
<br/>
<br/>
<br/>


 
## Use chain-of-thought to solve complicated problem
 
instead of input one single prompt, you can input multiple small stop questions to get better answer.  
 
 
```python
import akasha
import os


os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "boss/"
prompt = ["若陀龍王boss的詳細資訊", "若陀龍王boss的弱點是甚麼，該如何攻略","根據若陀龍王boss的弱點，幫我組出一隊四人的角色去攻略boss"]

response = akasha.chain_of_thought(dir_path, prompt)
print(response)
	
	
```


```python
根據若陀龍王的弱點，以下是一支適合攻略boss的四人隊伍建議：

1. 主DPS角色：選擇一位遠程輸出角色，如甘雨。她的遠距離射擊能夠完全發揮優勢，並且她的技能可以造成大量傷害。

2. 副DPS角色：選擇一位近戰輸出角色，如凱亞。凱亞的技能可以造成冰元素傷害，並且他的元素爆發可以提供額外的輸出。

3. 護盾角色：選擇一位能提供護盾的角色，如鐘離。鐘離的護盾厚度高且能夠提供額外的輸出，可以幫助隊伍減少受到的傷害。

4. 奶媽角色：選擇一位能提供回血的角色，如七七。七七的E技能可以持續回血，讓輸出位能夠持續站樁輸出。
```

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
response = akasha.get_response(dir_path, prompt,record_exp=exp_name)

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


## Test Performance

To evaluate the performance of current parameters, you can use function **test_performance** . First you need to build a question set .txt
file based on the documents you want to use, each question in the question set must be a single choice question, every options and the correct
answer is separated by space, each line is a question, **for example:**  (question_pvc.txt)

```text

應回收廢塑膠容器材質種類不包含哪種?  聚丙烯（PP） 聚苯乙烯（PS） 聚氯乙烯（PVC）  低密度聚乙烯（LDPE）  4
庫存盤點包括庫存全盤作業及不定期抽盤作業，盤點計畫應包括下列項目不包含哪項?   盤點差異之處理   盤點清冊  各項物品存放區域配置圖  庫存全盤日期及參加盤點人員名單  1
以下和者不是環保署指定之公民營地磅機構?    中森加油站企業有限公司   台益地磅站  大眾地磅站  新福行  4

```

**test_performance** will return the correct rate of the question set, details of each question would save in logs, or in mlflow server if you
turn on **record_exp** 

```python
import akasha
import os
from dotenv import load_dotenv
load_dotenv() 

os.environ["OPENAI_API_KEY"] = "your openAI key"
dir_path = "doc/pvc/"
exp_name = "exp_akasha_test_performance"

print(akasha.test_performance("question_pvc.txt", dir_path, search_type='merge',\
     model="openai:gpt-3.5-turbo", embeddings="openai:text-embedding-ada-002",record_exp=exp_name))
## 1.0 ##
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
import akasha
import os
from dotenv import load_dotenv
load_dotenv() 

os.environ["OPENAI_API_KEY"] = "your openAI key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your huggingface key"
dir_path = "doc/pvc/"
exp_name = "exp_akasha_optimum_combination"
embeddings_list = ["hf:shibing624/text2vec-base-chinese", "openai:text-embedding-ada-002"]
model_list = ["openai:gpt-3.5-turbo","hf:FlagAlpha/Llama2-Chinese-13b-Chat-4bit","hf:meta-llama/Llama-2-7b-chat-hf",\
            "llama-gpu:model/llama-2-7b-chat.ggmlv3.q8_0.bin", "llama-gpu:model/llama-2-13b-chat.ggmlv3.q8_0.bin"]
akasha.optimum_combination("question_pvc.txt", dir_path, embeddings_list = embeddings_list,model_list = model_list,
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
