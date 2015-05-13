# build from ubuntu
FROM ubuntu:14.04
MAINTAINER Kyle F <kylef@lab41.org>

# reset working directory
ENV DEBIAN_FRONTEND noninteractive

# update
RUN apt-get update

# install utilities
RUN apt-get install --assume-yes  vim less net-tools inetutils-ping curl git telnet nmap socat dnsutils netcat tree htop unzip sudo software-properties-common

# install python foundation
RUN apt-get install --assume-yes  build-essential python-dev python-pip liblapack-dev libatlas-dev gfortran libfreetype6 libfreetype6-dev libpng12-dev python-lxml libyaml-dev g++ libffi-dev

# update ipython
RUN apt-get remove --purge --assume-yes ipython && \
    pip install ipython jinja2 tornado jsonschema terminado
ENV IPYTHONDIR /ipython
RUN mkdir /ipython && \
    ipython profile create nbserver

# download/install neon
RUN cd /opt/ && git clone https://github.com/NervanaSystems/neon.git
RUN cd /opt/neon && make install
WORKDIR /opt/neon

# default to shell
CMD ["/bin/bash"]
