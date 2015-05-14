#!/bin/bash

__image=lab41/neon

docker run  -it \
            --privileged \
            --volume $(pwd)/data:/root/data \
            $__image \
              bash
