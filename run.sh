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

docker run -d -p 8000:8000 -v $NLTK_DATA:/root/nltk_data \
                           -v $MODELS:/thingin_Adapter/Semantic_Search/models \
                           -v $CACHES:/thingin_Adapter/recommender/cache_files \
                           --mount type=bind,source=$LOG,target=/thingin_Adapter/thingin_recommender.log \
                           thingin_adapter:v1

