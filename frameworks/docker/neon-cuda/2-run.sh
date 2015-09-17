#!/bin/bash

__image=lab41/neon-cuda

docker run  -it \
            --privileged \
            --volume $(pwd):/root/data \
            $__image \
              bash
