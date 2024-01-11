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



<br/>

(optional) if you want to use.gguf models or other models from huggingface, you can download the model and put it in './model' folder in the same directory of api.py, then you can use it in consult page.



# Docker 
To use Docker to run akasha dev-ui, you can clone the whole project and use Dockerfile to build the image 

``` bash
git clone https://gitlab-devops.iii.org.tw/root/qaiii-1.git@dev-ui
mkdir model  # you can put the model you want to use in here
mkdir config # directory that save the dataset, expert configs
mkdir docs # directory that save the document files
sudo docker build -t akasha_dev_ui:0.1 .
sudo docker run -v ./model:/app/model -v ./config:/app/config -v ./docs:/app/docs -p 8501:8501 --name akasha_dev_ui akasha_dev_ui:0.1 



```