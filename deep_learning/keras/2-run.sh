#!/bin/bash

# image name
__image=lab41/keras
__volume_host=/opt
__volume_cntr=/data

# run image
docker run -it \
           --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
           --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
           --volume=$__volume_host:$__volume_cntr \
            $__image
