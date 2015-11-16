#!/bin/bash

# image name
__image=lab41/neon-cuda7.5

# build image
docker build -t $__image .
