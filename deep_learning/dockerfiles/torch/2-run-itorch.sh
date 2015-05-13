#!/bin/bash

# image name
__image=lab41/torch

# run image
docker run  -d \
            -P \
            $__image \
              itorch notebook --ip='*'
