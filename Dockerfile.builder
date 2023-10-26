FROM python:3.8-slim
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y git gcc-11 build-essential g++ clang
RUN python -m pip install --upgrade pip && \
python -m pip install -r requirements.txt
RUN python -m pip install streamlit
RUN python -m pip install streamlit-option-menu