FROM python:3.8-slim
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY api.py /app
COPY requirements.txt /app
COPY start.sh /app
COPY api_utils.py /app
COPY main.py /app
COPY utils.py /app
#COPY accounts.yaml /app
COPY ./views /app/views
COPY ./routers /app/routers
RUN chmod u+x *.sh
RUN apt-get update && apt-get install -y gcc-11 build-essential g++ clang curl
RUN python -m pip install --upgrade pip && \
python -m pip install -r requirements.txt 
EXPOSE 8501
EXPOSE 8000
ENTRYPOINT nohup /bin/bash -c "./start.sh &" && streamlit run main.py --server.maxUploadSize 200000  --server.port 8501