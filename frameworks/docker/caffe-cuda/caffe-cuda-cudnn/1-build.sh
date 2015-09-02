#!/bin/bash
set -e

if ! ls cudnn*.tgz 1>/dev/null 2>&1; then
    echo "cuDNN must be downloaded seperately and placed in this directory"
    exit 1
fi    
# image name
__image_base=lab41/caffe-cuda-base
__image=lab41/caffe-cuda-cudnn

# build image
echo "Building caffe-cuda-base"
pushd ../caffe-cuda-base
docker build -t $__image_base .
popd
echo "Building caffe-cuda-cudnn"
docker build -t $__image .

