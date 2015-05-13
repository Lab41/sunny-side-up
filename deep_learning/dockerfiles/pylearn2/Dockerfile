# build off pylearn2 dockerfile
FROM tleyden5iwx/pylearn2
MAINTAINER Kyle F <kylef@lab41.org>

# configure environment
RUN mkdir --parents /data/shared
ENV PYLEARN2_DATA_PATH /data/shared
VOLUME /data/shared
WORKDIR /opt/pylearn2/pylearn2/scripts/tutorials/grbm_smd

# enable ipython notebook
EXPOSE 8888

# default to ipython notebook
CMD ["ipython","notebook","--ip='*'"]
