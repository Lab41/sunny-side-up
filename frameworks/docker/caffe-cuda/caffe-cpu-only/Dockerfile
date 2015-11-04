FROM lab41/caffe-cuda-base
MAINTAINER Yonas Tesfaye <yonast@lab41.org>

# Clone Caffe repo and move into it
RUN cd /root/caffe && \
  cat Makefile.config.example | \
  sed 's!# ANACONDA_HOME := $(HOME)/anaconda!ANACONDA_HOME := /opt/conda!' | \
  sed 's!# $(ANACONDA_HOME)! $(ANACONDA_HOME)!' | \
  sed 's!# PYTHON_INCLUDE := $(ANACONDA_HOME)!PYTHON_INCLUDE := $(ANACONDA_HOME)!'  >  /root/caffe/Makefile.config && \
  sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config && \
  make clean && \
  make -j"$(nproc)" all && \
  make pycaffe

ENV PYTHONPATH /root/caffe/python

EXPOSE 5000

WORKDIR /root/caffe
