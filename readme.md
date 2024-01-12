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

<br/>
<br/>

1. use git to clone or download the project 

``` bash
git clone https://gitlab-devops.iii.org.tw/root/qaiii-1.git@dev-ui

```


2. (optional) edit install.env file 

(install.env)

``` bash
MODEL=./model    # put the model you want to use in here
CONFIG=./config # directory that save the dataset, expert configs
DOCS=./docs # directory that save the document files
IMAGE_NAME=akasha_dev_ui
IMAGE_VERSION=0.1

```


3. run the script to build image and run the container

``` bash 
sudo bash install.sh

```





# Run Docker image

you can download the docker image in [akasha_dev_ui.tar](https://iiiorgtw-my.sharepoint.com/:u:/g/personal/ccchang_iii_org_tw1/Eey7F7wIlldNrqiwdWE1H7wBb-TMCv3NY4rYxRUP5DVHug?e=OYwvkx) 


``` bash 
sudo docker load -i akasha_dev_ui.tar
sudo docker run -v ./model:/app/model -v ./config:/app/config -v ./docs:/app/docs -v ./chromadb:/app/chromadb -p 8501:8501 --name akasha_dev_ui akasha_dev_ui:0.1 

```