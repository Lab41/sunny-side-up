# build from ubuntu
FROM ubuntu:14.04
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

# install/configure CUDA
RUN cd  /tmp && \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb && \
    apt-get update && \
    apt-get install --assume-yes cuda
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib:$LD_LIBRARY_PATH
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

# download/configure/install neon
RUN cd /opt/ && git clone https://github.com/NervanaSystems/neon.git
RUN cd /opt/neon && \
    make sysinstall

# make some updates
RUN pip install -U six

# default to shell in root dir
WORKDIR /root
CMD ["/bin/bash"]
