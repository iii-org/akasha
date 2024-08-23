FROM ccchang0518/akasha-lab-builder:0.5
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
EXPOSE 8000
RUN python -m pip install streamlit==1.32.0 akasha-terminal==0.8.57

ENV PORT 8501
ENV PREFIX akasha-lab
ENV USE_PREFIX false
ENV ANONYMIZED_TELEMETRY false
EXPOSE $PORT
CMD if [ "$USE_PREFIX" = "true" ]; then \
    nohup /bin/bash -c "./start.sh &" && streamlit run main.py --server.maxUploadSize 200000  --server.port 8501 --browser.serverAddress 0.0.0.0 --server.headless true --server.baseUrlPath /${PREFIX}/; \
else \
    nohup /bin/bash -c "./start.sh &" && streamlit run main.py --server.maxUploadSize 200000  --server.port 8501 --browser.serverAddress 0.0.0.0 --server.headless true ; \
fi
#ENTRYPOINT nohup /bin/bash -c "./start.sh &" && streamlit run main.py --server.maxUploadSize 200000  --server.port 8501 --browser.serverAddress 0.0.0.0 --server.headless true --server.baseUrlPath ${baseUrl}