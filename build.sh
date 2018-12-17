#!/usr/bin/env bash

if [ ! -f build.sh ]; then
    echo "please go to the thingin adapter root folder, then execute!"
fi

docker build -t thingin_adapter:v1 .