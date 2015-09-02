FROM lab41/caffe-cuda-base
MAINTAINER Yonas Tesfaye <yonast@lab41.org>

# Clone Caffe repo and move into it
RUN cd /root/caffe && \
  cat Makefile.config.example | \
  sed 's!# ANACONDA_HOME := $(HOME)/anaconda!ANACONDA_HOME := /opt/conda!' | \
  sed 's!# $(ANACONDA_HOME)! $(ANACONDA_HOME)!' | \
  sed 's!# PYTHON_INCLUDE := $(ANACONDA_HOME)!PYTHON_INCLUDE := $(ANACONDA_HOME)!'  >  /root/caffe/Makefile.config && \
  make clean && \
  make -j"$(nproc)" all && \
  make pycaffe

ENV PYTHONPATH /root/caffe/python

WORKDIR /root/caffe

EXPOSE 8888

CMD ["/opt/conda/bin/ipython", "notebook", "--ip='*'", "--no-browser"]