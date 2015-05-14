#!/bin/bash

# image name
__image=lab41/neon-cuda

# build image
docker build -t $__image .
