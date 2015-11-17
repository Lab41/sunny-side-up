FROM lab41/caffe-cuda-base
MAINTAINER Yonas Tesfaye <yonast@lab41.org>

# Move cuDNN over and extract
ADD cudnn-*.tgz /root/

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

ENV THEANO_FLAGS mode=FAST_RUN,device=gpu,floatX=float32

RUN pip install keras && conda install -y Theano

WORKDIR /root/caffe

EXPOSE 8888

#CMD ["/opt/conda/bin/ipython", "notebook", "--ip='*'", "--no-browser"]
