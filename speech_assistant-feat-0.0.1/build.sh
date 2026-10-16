#!/bin/bash
set -e
imgname="harbor-contest.4pd.io/wanglimin/judge-flow/speech-assistant-darvin:0.0.1"
docker build --network=host -t $imgname .
docker push $imgname
# docker rmi -f $imgname
