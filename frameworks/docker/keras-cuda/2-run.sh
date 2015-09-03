#!/bin/bash

# image name
__image=lab41/keras-cuda
__volume_host=$(pwd)
__volume_cntr=/data
__port_host=80
__port_cntr=8888

# run image
docker run -it \
           --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
           --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
           --env THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" \
           --volume=$__volume_host:$__volume_cntr \
           --publish=$__port_host:$__port_cntr \
            $__image
