# build on set of datasci tools
FROM lab41/python-datatools
MAINTAINER Kyle F <kylef@lab41.org>

# upgrade numpy
RUN pip install --upgrade numpy

# add jupyter notebook
RUN pip install jupyter

# add gensim and glove
RUN pip install gensim glove

# add glove-python
RUN cd /tmp && \
    git clone https://github.com/maciejkula/glove-python.git && \
    cd glove-python && \
    python setup.py install

# upgrade six for custom glove code
RUN pip install six==1.9.0

# install custom glove code
RUN cd /tmp && \
    git clone https://github.com/danforth36phd/sunny-side-up.git && \
    cd sunny-side-up/src/glove && \
    python setup.py install

# setup data volume
VOLUME ["/data"]
WORKDIR /data

# default to jupyter notebook
ADD config/notebook.sh /notebook.sh
CMD ["/notebook.sh"]
