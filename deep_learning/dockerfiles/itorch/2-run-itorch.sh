#!/bin/bash

# image name
__image=lab41/itorch

# run image
docker run  -d \
            -P \
            --volume=$(pwd)/tutorials:/data \
            $__image
