#!/bin/bash

# image name
__image=lab41/caffe-cuda-cudnn
__volume_host=$1
__volume_cntr=/data
__ipython_port=$2

# run image
docker run -it\
	--device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        --device /dev/nvidia0:/dev/nvidia0 \
        --volume=$__volume_host:$__volume_cntr \
        -p $__ipython_port:8888 \
        $__image
