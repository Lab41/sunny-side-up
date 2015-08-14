#!/bin/bash

# image name
__ip=<IP-OF-MACHINE>
__image=lab41/itorch-cuda
__port_host=80
__port_cntr=8888
__volume_host=/opt/sunny-side-up
__volume_cntr=/data

# run image
docker run  -d \
            --publish=$__port_host:$__port_cntr \
            --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
            --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
            --volume=$__volume_host:$__volume_cntr \
            $__image

echo "Visit iTorch notebook at http://$__ip:$__ip_host"
