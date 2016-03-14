#!/bin/bash

# image name
__image=lab41/sentiment-ml

# volume mounts
__volume_host=/opt/sunny-side-up
__volume_cntr=/data

# choose tokenizer [nltk|stanford]
ARABIC_TOKENIZER=nltk

# run image
docker run -it \
           --env="DISPLAY=$DISPLAY" \
           --env="ARABIC_TOKENIZER=$ARABIC_TOKENIZER" \
           --volume=/tmp/.X11-unix:/tmp/.X11-unix \
           --volume=$__volume_host:$__volume_cntr \
            $__image bash
