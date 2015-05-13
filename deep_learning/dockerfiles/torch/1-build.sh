#!/bin/bash

# image name
__image=lab41/torch

# build image
docker build -t $__image .
