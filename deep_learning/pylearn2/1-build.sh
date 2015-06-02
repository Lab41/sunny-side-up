#!/bin/bash

# image name
__image=lab41/pylearn2

# build image
docker build -t $__image .
