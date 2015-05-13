#!/bin/bash

__image=lab41/neon

docker run  -it \
            --volume $(pwd)/data:/root/data \
            $__image \
              bash
