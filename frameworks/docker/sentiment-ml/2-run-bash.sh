#!/bin/bash

# image name
__image=lab41/sentiment-ml

# volume mounts
__volume_host=/opt/sunny-side-up
__volume_cntr=/data

# run image
docker run -it \
           --volume=$__volume_host:$__volume_cntr \
            $__image bash
