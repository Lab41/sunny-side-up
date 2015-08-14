#!/bin/bash

# image name
__image=lab41/itorch

# build image
docker build -t $__image .
