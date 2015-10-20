#!/bin/bash

# image name
__image=lab41/mechanical-turk

# build image
docker build -t $__image .
