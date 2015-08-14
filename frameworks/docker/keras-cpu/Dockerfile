# keras builds on theano
FROM kaixhin/theano
MAINTAINER Kyle F <kylef@lab41.org>


#TODO install HDFS?
RUN apt-get update
RUN apt-get install --assume-yes  python-dev \
                                  libhdf5-dev

# install python modules
RUN pip install numpy \
                scipy \
                cython \
                h5py \
                pyyaml \
                six==1.9.0


# install keras
RUN cd /tmp && \
    git clone https://github.com/fchollet/keras.git && \
    cd keras && \
    python setup.py install && \
    rm -rf /tmp/keras


# Adding ipython
RUN pip install ipython \
                pyzmq \
                jinja2 \
                tornado \
                jsonschema


# setup data volume
VOLUME ["/data"]
WORKDIR /data


# default to shell
CMD ["/bin/bash"]
