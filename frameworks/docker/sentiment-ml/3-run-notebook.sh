#!/bin/bash

# image name
__image=lab41/sentiment-ml

# volume mounts
__volume_host=/opt/sunny-side-up
__volume_cntr=/data

# exposed ports
__port_host=8888
__port_cntr=8888

# run image
docker run -d \
           --volume=$__volume_host:$__volume_cntr \
           --publish=$__port_host:$__port_cntr \
            $__image
