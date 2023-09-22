FROM python:3.8-slim
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY ui.py /app
COPY ./interface /app/interface
RUN apt-get update && apt-get install -y git gcc-11 build-essential g++ clang
RUN python -m pip install --upgrade pip && \
python -m pip install git+https://gitlab-devops.iii.org.tw/root/qaiii-1.git
RUN python -m pip install streamlit
RUN python -m pip install streamlit-option-menu
COPY ./docs/mic /app/docs/mic
#COPY ./model /app/model
EXPOSE 8501

# run container example : sudo docker run -p 8501:8501 -v ./akasha_ui/model:/app/model  --gpus all -it akasha_ui:1.0
ENTRYPOINT streamlit run ui.py --server.port 8501