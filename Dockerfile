FROM ccchang0518/akasha-lab-builder
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY api.py /app
COPY start.sh /app
COPY api_utils.py /app
COPY main.py /app
COPY utils.py /app
#COPY accounts.yaml /app
COPY ./views /app/views
COPY ./routers /app/routers
RUN chmod u+x *.sh
EXPOSE 8501
EXPOSE 8000
ENTRYPOINT nohup /bin/bash -c "./start.sh &" && streamlit run main.py --server.maxUploadSize 200000  --server.port 8501