FROM akasha_builder:1.0
LABEL chih-chuan chang<ccchang@iii.org.tw>

WORKDIR /app
COPY ui.py /app
COPY ./interface /app/interface
COPY ./akasha /app/akasha

#RUN python -m pip install git+https://gitlab-devops.iii.org.tw/root/qaiii-1.git
COPY ./docs/mic /app/docs/mic
#COPY ./model /app/model
EXPOSE 8501

# run container example : sudo docker run -p 8501:8501 -v ./akasha_ui/model:/app/model  --gpus all -it akasha_ui:1.0
ENTRYPOINT streamlit run ui.py --server.port 8501