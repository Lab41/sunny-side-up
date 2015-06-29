#!/bin/bash

# image name
__image=lab41/keras

# build image
docker build -t $__image .
