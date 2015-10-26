#!/bin/bash

# image name
__image=lab41/sentiment-ml

# build image
docker build -t $__image .
