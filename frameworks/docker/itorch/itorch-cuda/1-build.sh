#!/bin/bash

# image name
__image=lab41/itorch-cuda

# build image
docker build -t $__image .
