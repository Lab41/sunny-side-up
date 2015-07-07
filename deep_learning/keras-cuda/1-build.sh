#!/bin/bash

# image name
__image=lab41/keras-cuda

# build image
docker build -t $__image .
