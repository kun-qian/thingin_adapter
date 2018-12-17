#!/usr/bin/env bash


BASE=$(dirname "$PWD")"/mymodel/"
NLTK_DATA=$BASE"nltk_data"
MODELS=$BASE"models"
CACHES=$BASE"cache_files"
LOG=$BASE"thingin_recommender.log"

#echo $BASE
#echo $NLTK_DATA
#echo $MODELS
#echo $CACHES
#echo $LOG

if [[ ! "$(docker ps -a -f name=thingin_adapter)" ]]; then
    if [[ "$(docker ps -aq -f status=exited -f name=thingin_adapter)" ]]; then
        # cleanup
        echo "container exited, remove the container"
        docker rm thingin_adapter
    fi
else
    echo "container is running, stop and remove it"
    docker stop "$(docker ps -aq -f name=thingin_adapter)"
    docker rm thingin_adapter

echo "run the container"
docker run -d -p 8000:8000 -v $NLTK_DATA:/root/nltk_data \
                           -v $MODELS:/thingin_Adapter/Semantic_Search/models \
                           -v $CACHES:/thingin_Adapter/recommender/cache_files \
                           --mount type=bind,source=$LOG,target=/thingin_Adapter/thingin_recommender.log \
                           --name thingin_adapter
                           thingin_adapter:v1
fi
