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
## Select different embeddings

Using parameter "embeddings", you can choose different embedding models, and the embedding model will be used to store documents into vector storage and search relevant documents from prompt.  Default is **openai:text-embedding-ada-002**.

Currently support **openai**, **huggingface** and **tensorflowhub**.

### huggingface example

``` 
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2")
```
To use huggingface embedding models, you can type huggingface:model_name or hf:model_name, for example, **huggingface:all-MiniLM-L6-v2**


## Select different models
Using parameter "model", you can choose different text generation models, default is **openai:gpt-3.5-turbo**.

Currently support **openai**, **llama-cpp** and **huggingface**.

### 1.huggingface example
``` python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="hf:meta-llama/Llama-2-13b-chat-hf")

```

To use text generation model from **huggingface**, for example, meta llama, you can type **hf:meta-llama/Llama-2-13b-chat-hf**

### 2.llama-cpp example
llama-cpp can use quantized llama model and run on cpu, after you download or transfer llama-cpp model file using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

```python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="hf:meta-llama/llama-cpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin")

```

For example, if q4 model is in the "model/" directory, you can assign **llama-cpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin** to load model.

```python
akasha.get_response(dir_path, prompt, embeddings="huggingface:all-MiniLM-L6-v2", model="hf:meta-llama/llama-gpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin")

```
you can also combine gpu with cpu to run llama-cpp, using **llama-gpu:model/llama-2-13b-chat.ggmlv3.q4_0.bin**
 
 
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
## use AiiDO to record experiment 

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
exp_name = "exp_akasha_gr"
response = akasha.get_response(dir_path, prompt,record_exp=exp_name)

```
<br/>
<br/>

![image](https://gitlab-devops.iii.org.tw/root/qaiii-1/-/blob/master/pic/upload_experiments.png)


<br/>
<br/>


you can also compare the responses from different models, search type and embeddings

![image](https://gitlab-devops.iii.org.tw/root/qaiii-1/-/blob/master/pic/response_comparison.png)
