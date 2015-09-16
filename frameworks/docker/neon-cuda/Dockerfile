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
                terminado

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

# download and install nervana fork of cuda-convnet2
RUN cd /opt && git clone https://github.com/NervanaSystems/cuda-convnet2.git
RUN cd /opt/cuda-convnet2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install
ENV LD_LIBRARY_PATH /opt/cuda-convnet2/build/cudanet:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /opt/cuda-convnet2/build/cudanet:$LIBRARY_PATH

# download/install nervana imgworker
RUN cd /opt && git clone https://github.com/NervanaSystems/imgworker.git
RUN cd /opt/imgworker && \
    pip install .

# download/configure/install neon
RUN pip install virtualenv
RUN cd /opt/ && git clone https://github.com/NervanaSystems/neon.git
ENV NEON_CONFIG_CPU 1
ENV NEON_CONFIG_GPU 1
ENV NEON_CONFIG_MPI 0
ENV NEON_CONFIG_DEV 0
RUN cd /opt/neon && \
    sed -i "s/^CPU = .*/CPU = $NEON_CONFIG_CPU/g" setup.cfg && \
    sed -i "s/^GPU = .*/GPU = $NEON_CONFIG_GPU/g" setup.cfg && \
    sed -i "s/^DIST = .*/DIST = $NEON_CONFIG_MPI/g" setup.cfg && \
    sed -i "s/^DEV = .*/DEV = $NEON_CONFIG_DEV/g" setup.cfg && \
    make sysinstall

# make some updates
RUN pip install -U six

# default to shell in neon dir
WORKDIR /opt/neon
CMD ["/bin/bash"]
