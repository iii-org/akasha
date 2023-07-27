## OpenAI example


``` 
import QAIII
import os
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亚花」的花语是什么?	「失之交臂的感情」	「赤诚的心」	「浪子的真情」	「无法挽回的爱」"
with get_openai_callback() as cb:
	response = QAIII.get_response(dir_path, prompt)
	print(response)
    print(cb.total_tokens, cb.total_cost)
```