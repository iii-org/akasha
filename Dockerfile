FROM ccchang0518/akasha-lab-builder
LABEL chih-chuan chang<ccchang@iii.org.tw>

# Set up a non-root user
ARG USER=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USER} \
    && useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USER}

# Set the default user
USER ${USER}

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
