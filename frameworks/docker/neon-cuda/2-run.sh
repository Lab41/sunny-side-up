#!/bin/bash

# choose Docker image to use
__image=lab41/neon-cuda

# put together docker cmd with variable number of devices
CMDSTR="docker run  -it \
                    --volume ${pwd}:/root/data \
                    --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"

# attach GPU devices
for GPUDEV in $(ls /dev/nvidia[0-9]*); do
    CMDSTR="${CMDSTR} --device ${GPUDEV}:${GPUDEV}"
done

# specify image (and optional) command
CMDSTR="${CMDSTR} '$__image' bash"

# start container
echo -e "evaluating $CMDSTR"
eval "$CMDSTR"
