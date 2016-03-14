# build from existing torch dockerfile
FROM kaixhin/cuda-torch
MAINTAINER Kyle F <kylef@lab41.org>

# update
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update

# install utilities
RUN apt-get install --assume-yes  vim \
                                  less \
                                  net-tools \
                                  inetutils-ping \
                                  curl \
                                  git \
                                  telnet \
                                  nmap \
                                  socat \
                                  dnsutils \
                                  netcat \
                                  tree \
                                  htop \
                                  unzip \
                                  sudo \
                                  software-properties-common \
                                  wget

# install python foundation
RUN apt-get install --assume-yes  build-essential \
                                  python-dev \
                                  python-pip \
                                  liblapack-dev \
                                  libatlas-dev \
                                  gfortran \
                                  libfreetype6 \
                                  libfreetype6-dev \
                                  libpng12-dev \
                                  python-lxml \
                                  libyaml-dev \
                                  g++ \
                                  libffi-dev

# install lua foundation
RUN apt-get install --assume-yes lua5.2 libzmq3-dev lua5.2-dev
RUN cd /tmp && \
    wget http://luarocks.org/releases/luarocks-2.2.2.tar.gz && \
    tar zxpf luarocks-2.2.2.tar.gz && \
    cd luarocks-2.2.2 && \
    ./configure && \
    make && \
    make install

# update ipython
ENV testvar 2
RUN apt-get remove --purge --assume-yes ipython && \
    pip install ipython jinja2 tornado jsonschema terminado jupyter
ENV IPYTHONDIR /ipython
RUN mkdir /ipython && \
    ipython profile create nbserver

# install itorch dependencies and packages
RUN sudo apt-get install --assume-yes libssl-dev
RUN luarocks install luacrypto
RUN luarocks install uuid
RUN luarocks install lzmq
RUN luarocks install itorch
RUN luarocks install unsup
RUN luarocks install graph
RUN luarocks install nngraph

# download tutorials
RUN git clone https://github.com/torch/tutorials.git

# expose iTorch notebook
EXPOSE 8888
VOLUME ["/data"]
WORKDIR /data

# default iTorch notebook server
CMD ["itorch","notebook","--ip='*'"]
