#!/bin/bash

# image name
__image=lab41/pylearn2

# build image
docker run  -d \
            -P \
            --volume $(pwd)/data:/data/shared \
            $__image
