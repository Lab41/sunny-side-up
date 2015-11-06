#!/bin/bash

# name Docker imagee 
__image=lab41/neon-cuda

# put together docker cmd with variable number of devices
CMDSTR="docker run -it --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm "
# device 0
if [ -e '/dev/nvidia0' ]; then
   CMDSTR="${CMDSTR} --device /dev/nvidia0:/dev/nvidia0 "
fi
# device 1
if [ -e '/dev/nvidia1' ]; then
   CMDSTR="${CMDSTR} --device /dev/nvidia1:/dev/nvidia1 "
fi
# mount point, image, and command
CMDSTR="${CMDSTR} --volume $(pwd):/root/data '$__image' bash"

eval "$CMDSTR"

