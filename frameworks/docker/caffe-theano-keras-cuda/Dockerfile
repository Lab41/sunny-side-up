FROM lab41/keras-cuda
MAINTAINER Karl Ni

# Resetting home directory back to initial "/"
ENV HOME /

# From kaixhin/cuda-caffe
###################################
# Install git, bc and dependencies
RUN apt-get update && apt-get install -y \
  git \
  bc \
  libatlas-base-dev \
  libatlas-dev \
  libboost-all-dev \
  libopencv-dev \
  libprotobuf-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  protobuf-compiler \
  libhdf5-dev \
  libleveldb-dev \
  liblmdb-dev \
  libsnappy-dev

# Clone Caffe repo and move into it
RUN cd /root && git clone https://github.com/BVLC/caffe.git && cd caffe && \
# Copy Makefile
  cp Makefile.config.example Makefile.config && \
# Make
  make -j"$(nproc)" all
# Set ~/caffe as working directory
WORKDIR /root/caffe

# From Yonas's caffe-cuda-base
###################################
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1

ENV HOME /root/

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    cd / && wget --quiet https://repo.continuum.io/archive/Anaconda-2.3.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda-2.3.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda-2.3.0-Linux-x86_64.sh  && \
    /opt/conda/bin/conda install --yes conda==3.14.1 && \
    /opt/conda/bin/conda install --yes protobuf && \
    mkdir /root/cuda

ENV PATH /opt/conda/bin:$PATH

# From Yonas's caffe-cuda-cudnn
###################################
# Move cuDNN over and extract
ADD cudnn-*.tgz /root/
RUN mkdir /root/cuda/include && \
  mv /root/cudnn-6.5-linux-x64-v2/*.h /root/cuda/include/ && \
  mkdir /root/cuda/lib64 && \
  mv /root/cudnn-6.5-linux-x64-v2/* /root/cuda/lib64/

# Install cuDNN
RUN cd /root/cuda && \
  cp include/*.h /usr/local/cuda/include/ && \
  cp --preserve=links lib64/*.so* /usr/local/cuda/lib64/ && \
  cp lib64/*.a /usr/local/cuda/lib64/

# Clone Caffe repo and move into it
RUN cd /root/caffe && \
  cat Makefile.config.example | \
  sed 's!# ANACONDA_HOME := $(HOME)/anaconda!ANACONDA_HOME := /opt/conda!' | \
  sed 's!# $(ANACONDA_HOME)! $(ANACONDA_HOME)!' | \
  sed 's!# PYTHON_INCLUDE := $(ANACONDA_HOME)!PYTHON_INCLUDE := $(ANACONDA_HOME)!' | \
  sed 's!# USE_CUDNN := 1!USE_CUDNN := 1!' >  /root/caffe/Makefile.config && \
  make clean && \
  make -j"$(nproc)" all && \
  make pycaffe

ENV PYTHONPATH /root/caffe/python

# For iPython Notebook
# EXPOSE 8888
# CMD ["/opt/conda/bin/ipython", "notebook", "--ip='*'", "--no-browser"]

# Extra functionality for Karl
###################################
RUN apt-get install -y vim  && \
  pip install keras
RUN bash -l -c "ln /dev/null /dev/raw1394 && touch /dev/testfile"
# ln -s /dev/null /dev/raw1394

