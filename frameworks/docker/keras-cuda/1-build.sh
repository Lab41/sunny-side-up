#!/bin/bash

# build base
__image=lab41/keras-cuda
docker build -t $__image .
