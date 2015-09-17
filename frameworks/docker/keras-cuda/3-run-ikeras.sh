#!/bin/bash

# image name
__image=lab41/keras-cuda-jupyter

# data volume
__volume_host=/data
__volume_cntr=/data

# source code
__volume_host_src=$__volume_host/sunny-side-up/src/notebooks
__volume_host_dst=$__volume_host/ipython-notebooks

# server port
__port_host=80
__port_cntr=8888


# ensure notebook links exist
if [ ! -h $__volume_host_dst ]; then
  sudo ln -s $__volume_host_src $__volume_host_dst
fi

# run image
docker run -d \
           --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
           --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
           --env THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" \
           --env-file=config/github/token \
           --volume=$__volume_host:$__volume_cntr \
           --publish=$__port_host:$__port_cntr \
            $__image
