# keras builds on theano
FROM kaixhin/cuda-theano
MAINTAINER Kyle F <kylef@lab41.org>


#TODO install HDFS?
RUN apt-get update
RUN apt-get install --assume-yes  python-dev \
                                  libhdf5-dev

# install python modules
RUN pip install numpy
RUN pip install scipy
RUN pip install cython
RUN pip install h5py


# install cuDNN
ADD config/cudnn-6.5-linux-x64-v2.tgz /tmp
RUN cd /tmp/cudnn-6.5-linux-x64-v2 && \
    cp cudnn.h /usr/local/cuda/include && \
    cp libcudnn* /usr/local/cuda/lib


# install keras
RUN cd /tmp && \
    git clone https://github.com/fchollet/keras.git && \
    cd keras && \
    python setup.py install && \
    rm -rf /tmp/keras

# Adding ipython
RUN pip install ipython pyzmq jinja2 tornado jsonschema

# setup data volume
VOLUME ["/data"]

# default to shell
CMD ["/bin/bash"]
