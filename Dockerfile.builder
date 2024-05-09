FROM python:3.9-slim
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY requirements.txt /app
RUN apt-get update && apt-get install -y gcc-11 build-essential g++ clang curl
RUN python -m pip install --upgrade pip && \
python -m pip install -r requirements.txt 
