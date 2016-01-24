#!/bin/bash

# image name
__image=lab41/keras-cuda
__volume_host=$(pwd)
__volume_cntr=/data

# put together docker cmd with variable number of devices
CMDSTR="docker run -it \
           --volume=$__volume_host:$__volume_cntr \
           --env-file=docker-env.sh \
           --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"

# attach GPU devices
for GPUDEV in $(ls /dev/nvidia[0-9]*); do
    CMDSTR="${CMDSTR} --device ${GPUDEV}:${GPUDEV}"
done

# specify image (and optional) command
CMDSTR="${CMDSTR} $__image"

# start container
echo -e "evaluating $CMDSTR"
eval "$CMDSTR"
