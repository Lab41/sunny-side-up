# -*- coding: utf-8 -*-
import argparse, sys, os
import urllib
import zipfile, gzip
import glob
import shutil

from glove import Glove

class ModelDownloader:

    def __init__(self, model_type):

        # dataset names and urls to download
        if (model_type == 'glove'):
            self.data_location = {  'wikipedia-2014':       {   'url':  'http://nlp.stanford.edu/data/glove.6B.zip',
                                                                'files':[   'glove.twitter.27B.25d.txt',
                                                                            'glove.twitter.27B.50d.txt',
                                                                            'glove.twitter.27B.200d.txt',
                                                                            'glove.twitter.27B.300d.txt' ] },
                                    'twitter-2b':           {   'url':  'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                                                'files':[   'glove.twitter.27B.25d.txt',
                                                                            'glove.twitter.27B.50d.txt',
                                                                            'glove.twitter.27B.100d.txt',
                                                                            'glove.twitter.27B.200d.txt' ] },
                                    'common-crawl-42b':     {   'url':  'http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                                                'files':[   'glove.42B.300d.txt' ] },
                                    'common-crawl-840b':    {   'url':  'http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                                                'files':[   'glove.840B.300d.txt' ] }
                                }

            # use glove downloader
            self.download_and_save_vectors = self.download_and_save_vectors_glove

        elif (model_type == 'word2vec'):
            self.data_location = {  'google-news':          {   'url':  'https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz',
                                                                'files':[   'GoogleNews-vectors-negative300.bin'] }
                                 }

            # use word2vec downloader
            self.download_and_save_vectors = self.download_and_save_vectors_word2vec

        else:
            print 'BAD MODEL!'


    @staticmethod
    def download_fullpath(outdir, datafile):
        return os.path.join(outdir, "{}.obj".format(datafile))


    def download_and_save(self, outdir, dataset=None, datafile=None, maxmegabytes=None):
        '''
            download and save pre-trained glove/word2vec models
        '''

        # available data
        datasets_available = self.data_location.keys()

        # download specific model
        file_fullpath = self.download_fullpath(outdir, datafile)
        if dataset in datasets_available and not os.path.isfile(file_fullpath):
            url = self.data_location[dataset]['url']
            self.download_and_save_vectors(url=url, outdir=outdir, datafile=datafile, maxmegabytes=maxmegabytes)



    def download_and_extract_file(self, url, outdir):
        '''
            download and extract file

            Args:
                url:    where to get the file
                outdir: where to save the file

            Returns:
                string: directory where downloaded file is saved
        '''

        # construct filenames
        filename_full = os.path.basename(url)
        filename_base, filename_ext = os.path.splitext(filename_full)

        # download file if necessary
        filename_save = "{}/{}".format(outdir, filename_full)
        if not os.path.isfile(filename_save):
            print("downloading {}...".format(filename_save))
            urllib.urlretrieve(url, filename_save)

        # extract file into file-specific output directory
        dirname_file = "{}/{}".format(outdir, filename_base)
        if not os.path.isdir(dirname_file):

            # create directory
            os.mkdir(dirname_file)

            # extract compressed file
            print("extracting {}...".format(filename_save))

            # extract zipfiles to dirname_file
            if filename_ext == '.zip':
                with zipfile.ZipFile(filename_save, 'r') as z:
                    z.extractall(dirname_file)

            # extract gzipped files
            elif filename_ext == '.gz':

                # extract to directory of the same name (to match zipfiles)
                filename_uncompressed = os.path.basename(os.path.splitext(filename_save)[0])
                filepath_uncompressed = os.path.join(dirname_file, filename_uncompressed)

                # gunzip using shutil for block copies
                with gzip.open(filename_save) as z:
                    with open(filepath_uncompressed, 'wb') as f:
                        shutil.copyfileobj(z, f)
            else:
                print "Bad extension: {}".format(filename_ext)

        # notify to location of file
        return dirname_file



    def download_and_save_vectors_word2vec(self, url, outdir, datafile, maxmegabytes=None):
        '''
            download and save pre-trained word2vec model
        '''

        # download file
        dirname_file = self.download_and_extract_file(url, outdir)

        # extract file
        file_in = "{}/{}".format(dirname_file, datafile)

        # build model for file (w2v already built; just move file)
        fullpath_out = self.download_fullpath(outdir, datafile)
        shutil.move(file_in, fullpath_out)

        # remove extracted directory
        shutil.rmtree(dirname_file)


    def download_and_save_vectors_glove(self, url, outdir, datafile=None, maxmegabytes=None):
        '''
            download and save pre-trained glove model
        '''

        # download file
        dirname_file = self.download_and_extract_file(url, outdir)

        # extract file
        file_in = "{}/{}.txt".format(dirname_file, datafile)

        # build output filename
        fullpath_out = self.download_fullpath(outdir, datafile)

        # catch memory exceptions
        try:

            # ensure file isn't too big
            filesize = os.path.getsize(file_in) / 1024 / 1024
            filesize_ok = (not maxmegabytes or filesize <= int(maxmegabytes))

            # download specific file and/or files under specific limit
            if filesize_ok:
                print("importing glove vectors from {}".format(file_in))
                model = Glove.load_stanford(file_in)

                # save model object to specified output directory
                print("saving glove model to {}...".format(fullpath_out))
                model.save_obj(fullpath_out)
            else:
                print("skipping file {}...".format(file_in))


        except MemoryError as e:
            print e.strerror

        # remove extracted directory
        shutil.rmtree(dirname_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Download, import, and save GloVe models'
                                                  'from Stanford'))
    parser.add_argument('--outdir', '-o', action='store',
                        required=True,
                        help='The output directory to download/save the models.')
    parser.add_argument('--dataset', '-d', action='store',
                        default='all',
                        help='The dataset to download')
    parser.add_argument('--maxmegabytes', '-m', action='store',
                        default=250,
                        help='The dataset to download')
    args = parser.parse_args()

    downloader = ModelDownloader('glove')
    downloader.download_and_save(outdir=args.outdir, dataset='wikipedia-2014', datafile='glove.6B.50d.obj', maxmegabytes=args.maxmegabytes)

    downloader = ModelDownloader('word2vec')
    downloader.download_and_save(outdir='/data', dataset='google-news', datafile='GoogleNews-vectors-negative300.bin', maxmegabytes=args.maxmegabytes)
