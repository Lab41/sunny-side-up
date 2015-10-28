#!/bin/bash

# image name
__image=lab41/keras-cpu
__volume_host=$(pwd)
__volume_cntr=/data

# run image
docker run -it \
           --volume=$__volume_host:$__volume_cntr \
           --publish=8888:8888 \
            $__image
