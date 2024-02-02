#!/bin/bash

# Load variables from install.env
sudo apt install -y dos2unix jq
sudo dos2unix *
sudo dos2unix install.env
source install.env



## create directories
sudo mkdir -p $CONFIG
sudo mkdir -p $MODEL
sudo mkdir -p $DOCS
sudo mkdir -p $CHROMADB

JSON_OBJ='{}'
# Check if DEFAULT_OPENAI_API_KEY exists and if so, add it to the JSON object
if [[ -n $DEFAULT_OPENAI_API_KEY ]]; then
    JSON_OBJ=$(echo $JSON_OBJ | jq --arg key "$DEFAULT_OPENAI_API_KEY" '. + {"openai_key": $key}')
fi

# Check if DEFAULT_AZURE_API_KEY and DEFAULT_AZURE_API_BASE exist and if so, add them to the JSON object
if [[ -n $DEFAULT_AZURE_API_KEY ]] && [[ -n $DEFAULT_AZURE_API_BASE ]]; then
    JSON_OBJ=$(echo $JSON_OBJ | jq --arg key "$DEFAULT_AZURE_API_KEY" --arg base "$DEFAULT_AZURE_API_BASE" '. + {"azure_key": $key, "azure_base": $base}')
fi

# Write the JSON object to default_key.json only if it's not an empty object
if [[ $JSON_OBJ != '{}' ]]; then
    echo $JSON_OBJ | sudo tee $CONFIG/default_key.json > /dev/null
fi



# ## build image
 sudo docker build -t $IMAGE_NAME:$IMAGE_VERSION .

# ## run container
 sudo docker run -v $MODEL:/app/model -v $CONFIG:/app/config -v $DOCS:/app/docs -v $CHROMADB:/app/chromadb -v ./accounts.yaml:/app/accounts.yaml -p 8501:8501 -p 8000:8000 --name $IMAGE_NAME $IMAGE_NAME:$IMAGE_VERSION
