# akasha dev-ui

# Installation
1. clone all the files in dev-ui branch to your directory
2. install akasha package in sdk branch (pip install git+https://gitlab-devops.iii.org.tw/root/qaiii-1.git@sdk)
3. run fast api server (uvicorn api:app), url should be http://127.0.0.1:8000
4. run streamlit (streamlit run main.py)


# Usage
1. signup or login
2. setting openai api key in setting page
3. create new dataset in datasets page
4. create expert in experts page
5. ask question in consult page
(optional) if you want to use.gguf models or other models from huggingface, you can download the model and put it in './model' folder in the same directory of api.py, then you can use it in consult page.