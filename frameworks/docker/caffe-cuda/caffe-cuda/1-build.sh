#!/bin/bash
set -e

# image name
__image_base=lab41/caffe-cuda-base
__image=lab41/caffe-cuda

# build image
echo "Building caffe-cuda-base"
pushd ../caffe-cuda-base
docker build -t $__image_base .
popd
echo "Building caffe-cuda"
docker build -t $__image .

