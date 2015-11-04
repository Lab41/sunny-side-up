#!/bin/bash

source "docker-env.sh"

# image name
__image=lab41/keras-cuda-jupyter

# run image
docker run -d \
           --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
           --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
           --volume=$VOL_HOST:$VOL_CNTR \
           --publish=$PORT_HOST:$PORT_CNTR \
           --env-file=docker-env.sh \
            $__image
