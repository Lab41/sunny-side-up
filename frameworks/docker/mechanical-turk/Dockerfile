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


## install text->png
#RUN apt-get install --assume-yes  python-dev \
#                                  libjpeg-dev \
#                                  libfreetype6-dev \
#                                  zlib1g-dev
#RUN ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/ && \
#    ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/ && \
#    ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/
#RUN pip install Pillow



## install pango
#RUN apt-get install --assume-yes libcairo2-dev libjpeg8-dev libpango1.0-dev libgif-dev build-essential g++ libcairomm-1.0-dev libgtk3.0-cil-dev python-cairo-dev libxext-dev libxt-dev

#RUN mkdir -p $HOME/src $HOME/tmp
#ENV TMPDIR "$HOME/tmp"
#ENV PATH "$HOME/bin:$PATH"
#ENV C_INCLUDE_PATH "$HOME/include:$C_INCLUDE_PATH"
#ENV LIBRARY_PATH "$HOME/lib:$LIBRARY_PATH"
#ENV LD_LIBRARY_PATH "$HOME/lib:$LD_LIBRARY_PATH"
#ENV CPPFLAGS "-I$HOME/include $CPPFLAGS"
#ENV LDFLAGS "-L$HOME/lib $LDFLAGS"
#ENV PKG_CONFIG_PATH "$HOME/lib/pkgconfig"
#ENV PYTHON python2.7
#RUN mkdir -p $HOME/lib/${PYTHON}

## =========================================
## pixman 0.24.4
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/pixman-0.24.4.tar.gz' && \
#    tar -xzf pixman-0.24.4.tar.gz && \
#    cd pixman-0.24.4 && \
#    ./configure --prefix=$HOME && \
#    make && \
#    make install

## =========================================
## cairo 1.12.0
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/cairo-1.12.0.tar.gz' && \
#    tar -xzf cairo-1.12.0.tar.gz && \
#    cd cairo-1.12.0 && \
#    ./configure --prefix=$HOME --enable-xlib && \
#    make && \
#    make install

## =========================================
## libffi 3.0.11
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/libffi-3.0.11.tar.gz' && \
#    tar -xzf libffi-3.0.11.tar.gz && \
#    cd libffi-3.0.11 && \
#    ./configure --prefix=$HOME && \
#    make && \
#    make install

## =========================================
## Glib 2.31.22
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/glib-2.31.22.tar.xz' && \
#    tar -xJf glib-2.31.22.tar.xz && \
#    cd glib-2.31.22 && \
#    PYTHON=${PYTHON} ./configure --prefix=$HOME && \
#    make && \
#    make install

## =========================================
## pango 1.30.0
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/pango-1.30.0.tar.xz' && \
#    tar -xJf pango-1.30.0.tar.xz && \
#    cd pango-1.30.0 && \
#    ./configure --prefix=$HOME --sysconfdir=$HOME/etc && \
#    make CFLAGS=-I$HOME/src/cairo-1.12.0 && \
#    make install

## =========================================
## pygobject 2.28.6
## =========================================
#RUN apt-get install --assume-yes libgtk2.0-0
#RUN cd $HOME/src && \
#    wget 'http://mirror.ryansanden.com/xfce-4.8/pygobject-2.28.6.tar.bz2' && \
#    tar -xjf pygobject-2.28.6.tar.bz2 && \
#    cd pygobject-2.28.6 && \
#    PYTHON=${PYTHON} ./configure --prefix=$HOME --disable-introspection && \
#    make && \
#    make install

## =========================================
## py2cairo 1.10.0
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://www.cairographics.org/releases/py2cairo-1.10.0.tar.bz2' && \
#    tar -xjf py2cairo-1.10.0.tar.bz2 && \
#    cd py2cairo-1.10.0 && \
#    $PYTHON ./waf configure --prefix=$HOME ; \
#    sed -i "s/for incstr.*/for incstr in conf.cmd_and_log(conf.env.PYTHON_CONFIG + ' --includes').strip().split():/g" /py2cairo-1.10.0/.waf-1.6.3-3c3129a3ec8fb4a5bbc7ba3161463b22/waflib/Tools/python.py && \
#    $PYTHON ./waf configure --prefix=$HOME && \
#    $PYTHON ./waf build && \
#    $PYTHON ./waf install

## =========================================
## pygtk 2.24.0
## =========================================
#RUN cd $HOME/src && \
#    wget 'http://ftp.gnome.org/pub/GNOME/sources/pygtk/2.24/pygtk-2.24.0.tar.bz2' && \
#    tar -xjf pygtk-2.24.0.tar.bz2 && \
#    cd pygtk-2.24.0 && \
#    ./configure --prefix=$HOME && \
#    make && \
#    make install

# default to shell
CMD ["/bin/bash"]
