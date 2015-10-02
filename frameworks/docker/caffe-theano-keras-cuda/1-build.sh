#!/bin/bash

# build base
__image=lab41/caffe-keras-theano-cuda
docker build -t $__image .

# __image=lab41/keras-cuda-jupyter
# docker build -f Dockerfile_jupyter -t $__image .

