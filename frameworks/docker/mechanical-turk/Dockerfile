# keras adds to existing data tools
FROM lab41/python-datatools
MAINTAINER Kyle F <kylef@lab41.org>


# add font repo
RUN wget http://packages.sil.org/sil.gpg -O /tmp/sil.gpg && \
    apt-key add /tmp/sil.gpg && \
    rm /tmp/sil.gpg && \
    echo "deb http://packages.sil.org/ubuntu $(lsb_release -c | cut -f2) main" | sudo tee -a /etc/apt/sources.list && \
    apt-get update


# add scheherazade font
RUN apt-get install --assume-yes  ttf-sil-scheherazade \
                                  fonts-sil-scheherazade


# install python modules
RUN pip install boto


# install pango text->png
RUN apt-get update && \
    apt-get install --assume-yes libpango1.0-dev


# default to shell
CMD ["/bin/bash"]
