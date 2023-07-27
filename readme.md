## OpenAI example


``` 
import QAIII
import os
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "your openAI key"

dir_path = "doc/"
prompt = "「塞西莉亞花」的花語是什麼?	「失之交臂的感情」	「赤誠的心」	「浪子的真情」	「無法挽回的愛」"
with get_openai_callback() as cb:
	response = QAIII.get_response(dir_path, prompt)
	print(response)
	print(cb.total_tokens, cb.total_cost)
	
```