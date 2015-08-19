#!/bin/bash

# image name
__image=lab41/caffe-cuda

# build image
docker build -t $__image .
