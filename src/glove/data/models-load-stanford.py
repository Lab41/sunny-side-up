# -*- coding: utf-8 -*-
import argparse, sys, os
import urllib
import zipfile
import glob

from glove import Glove

def glove_vector_download_and_save(url, outdir, maxmegabytes):

    # construct filenames
    filename_full = os.path.basename(url)
    filename_name = os.path.splitext(filename_full)[0]

    # create file-specific output directory
    dirname_file = "{}/{}".format(outdir, filename_name)
    if not os.path.isdir(dirname_file):
        os.mkdir(dirname_file)

    # download file
    filename_save = "{}/{}".format(dirname_file, filename_full)
    if not os.path.isfile(filename_save):
        print("downloading {}...".format(filename_save))
        urllib.urlretrieve(url, filename_save)

    # extract zip
    print("extracting {}...".format(filename_save))
    with zipfile.ZipFile(filename_save, "r") as z:
        z.extractall(dirname_file)

    # build model for each file
    file_pattern = "{}/*.txt".format(dirname_file)
    for file_glove_in in glob.glob(file_pattern):

        try:
            # ensure file isn't too big
            filesize = os.path.getsize(file_glove_in) / 1024 / 1024
            if filesize > maxmegabytes:
                print("skipping {}M file {}...".format(filesize, file_glove_in))

            else:

                # load vectors
                print("importing glove vectors from {}".format(file_glove_in))
                model = Glove.load_stanford(file_glove_in)

                # save model object
                file_glove_out = "{}.obj".format(os.path.splitext(file_glove_in)[0])
                print("saving glove model to {}...".format(file_glove_out))
                model.save_obj(file_glove_out)

                # delete extracted file
                os.remove(file_glove_in)

        except MemoryError as e:
            print e.strerror


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

    # dataset names and urls to download
    datasets = {    'wikipedia-2014':     'http://nlp.stanford.edu/data/glove.6B.zip',
                    'common-crawl-42b':   'http://nlp.stanford.edu/data/glove.42B.300d.zip',
                    'common-crawl-840b':  'http://nlp.stanford.edu/data/glove.840B.300d.zip',
                    'twitter-2b':         'http://nlp.stanford.edu/data/glove.twitter.27B.zip' }

    # download files
    if args.dataset == 'all':
        for name, url in datasets.iteritems():
            glove_vector_download_and_save(url, args.outdir, args.maxmegabytes)

    elif args.dataset in datasets.keys():
        glove_vector_download_and_save(datasets[args.dataset], args.outdir, args.maxmegabytes)

    else:
        print 'Input Error:\nWhen using --dataset, you must specify the dataset from <"{}">'.format('" | "'.join(k))
