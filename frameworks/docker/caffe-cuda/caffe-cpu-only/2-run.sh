#!/bin/bash

# image name
__image=lab41/caffe-cpu-only
__volume_host=$1
__volume_cntr=/data
__ipython_port=$2

# run image
docker run -it\
        --volume=$__volume_host:$__volume_cntr \
        -p $__ipython_port:5000 \
        $__image
