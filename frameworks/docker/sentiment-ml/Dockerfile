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
    git clone https://github.com/Lab41/sunny-side-up.git && \
    cd sunny-side-up/src/glove && \
    python setup.py install

# add hdf5
RUN apt-get install --assume-yes libhdf5-dev
RUN pip install h5py

# add open weibo processing
RUN pip install jieba pypinyin

# update to java8 for stanford NLP
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    add-apt-repository -y ppa:webupd8team/java && \
    apt-get update && \
    apt-get install -y oracle-java8-installer && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

# install jpype for python-java bindings
RUN cd /tmp && \
    git clone https://github.com/originell/jpype.git && \
    cd jpype && \
    sed -i "s/elif jc.isSubclass('java.util.Iterator').*/elif jc.isSubclass('java.util.Iterator') and (members.has_key('next') or members.has_key('__next__')):/g" /tmp/jpype/jpype/_jcollection.py && \
    python setup.py install

# install stanford NLP
RUN cd /usr/lib && \
    wget http://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip && \
    unzip stanford-parser-full-2015-04-20.zip && \
    rm stanford-parser-full-2015-04-20.zip
ENV STANFORD_PARSER_HOME /usr/lib/stanford-parser-full-2015-04-20

# setup NLTK parser
RUN python -m nltk.downloader punkt

# setup data volume
VOLUME ["/data"]
WORKDIR /data

# default to jupyter notebook
ADD config/notebook.sh /notebook.sh
CMD ["/notebook.sh"]
