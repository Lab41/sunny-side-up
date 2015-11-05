# build from ubuntu 15.04, matching host
FROM ubuntu:15.04
MAINTAINER Kyle F <kylef@lab41.org>

# configure headless install
ENV DEBIAN_FRONTEND noninteractive

# install base
RUN apt-get update && \
    apt-get install --assume-yes \
                        bc \
                        build-essential \
                        cmake \
                        curl \
                        dnsutils \
                        g++ \
                        g++-4.6 \
                        g++-4.6-multilib \
                        gcc-4.6 \
                        gcc-4.6-multilib \
                        gfortran \
                        git \
                        htop \
                        inetutils-ping \
                        less \
                        libatlas-base-dev \
                        libatlas-dev \
                        libboost-all-dev \
                        libffi-dev \
                        libfreeimage-dev \
                        libfreetype6 \
                        libfreetype6-dev \
                        libhdf5-serial-dev \
                        libjpeg-dev \
                        libjpeg62 \
                        liblapack-dev \
                        libleveldb-dev \
                        liblmdb-dev \
                        libopencv-dev \
                        libpng12-dev \
                        libprotobuf-dev \
                        libsnappy-dev \
                        libyaml-dev \
                        net-tools \
                        netcat \
                        nmap \
                        pkgconf \
                        protobuf-compiler \
                        python-dev \
                        python-lxml \
                        python-magic \
                        python-matplotlib \
                        python-numpy \
                        python-pip \
                        python-scipy \
                        socat \
                        software-properties-common \
                        sudo \
                        telnet \
                        tree \
                        unzip \
                        vim \
                        wget

# Use gcc 4.6
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-4.6 30 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-4.6 30 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 30 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 30

# install python modules
RUN pip install ipython \
                jinja2 \
                tornado \
                jsonschema \
                terminado \
                simplejson

# setup ipython
ENV IPYTHONDIR /ipython
RUN mkdir /ipython && \
    ipython profile create nbserver

# install/configure CUDA 7.5 (for driver version 352.39)
# Change to the /tmp directory
RUN cd /tmp && \
# Download run file
  wget -nv http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run && \ 
# Make the run file executable and extract
  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
# Install CUDA drivers (silent, no kernel)
  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
# Install toolkit (silent)  
  ./cuda-linux64-rel-*.run -noprompt && \
# Clean up
  rm -rf *

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib:$LD_LIBRARY_PATH
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

# download/configure/install neon 1.0.0
RUN pip install virtualenv
RUN cd /opt/ && \
    wget -nv https://github.com/NervanaSystems/neon/archive/v1.0.0.tar.gz
RUN cd /opt && tar -xzvf v1.0.0.tar.gz && \
    mv neon-1.0.0 neon
RUN cd /opt/neon && \
    make sysinstall

# make some updates
RUN pip install -U six

# Make kludgey edit to neon asserts
RUN cd /usr/local/lib/python2.7/dist-packages/neon/backends && \
    sed -i 's/(CRST, KRST)/(CRST ,)/g' layer_gpu.py

# default to shell in root dir
WORKDIR /root
CMD ["/bin/bash"]
