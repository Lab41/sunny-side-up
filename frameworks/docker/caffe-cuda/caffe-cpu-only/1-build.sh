#!/bin/bash
set -e

# image name
__image_base=lab41/caffe-cuda-base
__image=lab41/caffe-cpu-only

# build image
echo "Building caffe-cpu-only"
pushd ../caffe-cuda-base
docker build -t $__image_base .
popd
echo "Building caffe-cpu-only"
docker build -t $__image .

