#!/bin/bash

# image name
__image=lab41/itorch-cpu

# build image
docker build -t $__image .
