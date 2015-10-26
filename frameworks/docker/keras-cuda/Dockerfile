# keras builds on theano
FROM kaixhin/cuda-theano
MAINTAINER Kyle F <kylef@lab41.org>


# install base packages
RUN apt-get update
RUN apt-get install --assume-yes  python-dev libhdf5-dev libpng12-dev libfreetype6-dev libpng++-dev libfreetype6 libfreetype6-dev libpng12-dev pkg-config


# install python modules
RUN pip install numpy \
                scipy \
                cython \
                h5py \
                pyyaml \
                six==1.9.0 \
                pandas \
                passage \
                simplejson \
                matplotlib


# Adding ipython
RUN pip install ipython \
                pyzmq \
                jinja2 \
                tornado \
                jsonschema


# install CUDA
RUN cd /tmp && \
    wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
    chmod +x cuda_*_linux.run && \
    ./cuda_*_linux.run -extract=`pwd` && \
    ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
    ./cuda-linux64-rel-*.run -noprompt && \
    rm -rf *


# Add CUDA to path
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# install keras
RUN cd /tmp && \
    git clone https://github.com/fchollet/keras.git && \
    cd keras && \
    python setup.py install && \
    rm -rf /tmp/keras


# setup data volume
VOLUME ["/data"]
WORKDIR /data


# default to shell
CMD ["/bin/bash"]
