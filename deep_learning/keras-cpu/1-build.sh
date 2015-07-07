#!/bin/bash

# image name
__image=lab41/keras-cpu

# build image
docker build -t $__image .
