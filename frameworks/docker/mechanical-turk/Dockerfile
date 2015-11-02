# keras adds to existing data tools
FROM lab41/python-datatools
MAINTAINER Kyle F <kylef@lab41.org>


# add font repo
RUN wget http://packages.sil.org/sil.gpg -O /tmp/sil.gpg && \
    apt-key add /tmp/sil.gpg && \
    rm /tmp/sil.gpg && \
    echo "deb http://packages.sil.org/ubuntu $(lsb_release -c | cut -f2) main" | sudo tee -a /etc/apt/sources.list && \
    apt-get update


# add font and text->png utility
RUN apt-get install --assume-yes  ttf-sil-scheherazade \
                                  fonts-sil-scheherazade \
                                  libpango1.0-dev


# python modules: mechanical turk; image processing
RUN pip install boto pillow


# add MTurk command line tools
RUN cd / && \
    wget https://mturk.s3.amazonaws.com/CLTSource/aws-mturk-clt.tar.gz && \
    tar zxvf aws-mturk-clt.tar.gz && \
    rm aws-mturk-clt.tar.gz


# set java
ENV JAVA_HOME /opt/jdk/jdk1.7.0_67


# start in the /data directory
WORKDIR /data


# default to shell
CMD ["/bin/bash"]
