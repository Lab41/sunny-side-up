#!/bin/bash

# image name
__ip=<IP-OF-MACHINE>
__image=lab41/itorch
__ip_host=80
__ip_cntr=8888

# run image
docker run  -d \
            --publish=$__ip_host:$__ip_cntr \
            --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
            --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
            --volume=$(pwd)/tutorials:/data \
            $__image

echo "Visit iTorch notebook at http://$__ip:$__ip_host"
