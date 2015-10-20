#!/bin/bash

# image name
__image=lab41/mechanical-turk
__volume_host=$(pwd)
__volume_cntr=/data

# run image
docker run -it \
           --volume=$__volume_host:$__volume_cntr \
            $__image
