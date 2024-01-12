#!/bin/bash

# Load variables from master.env
sudo apt install -y dos2unix
sudo dos2unix *
sudo dos2unix .env
source .env



## create directories
sudo mkdir -p $CONFIG
sudo mkdir -p $MODEL
sudo mkdir -p $DOCS


## build image
sudo docker build -t $IMAGE_NAME:$IMAGE_VERSION .

## run container
sudo docker run -v $MODEL:/app/model -v $CONFIG:/app/config -v $DOCS:/app/docs -p 8501:8501 --name $IMAGE_NAME $IMAGE_NAME:$IMAGE_VERSION
