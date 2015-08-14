#!/bin/bash

# image name
__image=lab41/itorch-cpu
__ip_host=49888
__ip_cntr=8888

# run image
docker run  -d \
            --publish=$__ip_host:$__ip_cntr \
            --volume=$(pwd)/tutorials:/data \
            $__image

echo "Visit iTorch notebook at http://localhost:49888"
