# build on set of datasci tools
FROM lab41/python-datatools
MAINTAINER Kyle F <kylef@lab41.org>

# add gensim and glove
RUN pip install gensim glove

# add glove-python
RUN cd /tmp && \
    git clone https://github.com/maciejkula/glove-python.git && \
    cd glove-python && \
    python setup.py install

# add jupyter notebook
RUN pip install jupyter

# setup data volume
VOLUME ["/data"]
WORKDIR /data

# default to jupyter notebook
ADD config/notebook.sh /notebook.sh
CMD ["/notebook.sh"]
