#!/usr/bin/env python

import os
import tarfile
import random
import logging
logger=logging.getLogger(__name__)
from data_utils import get_file

pos_label = 1
neg_label = 0

def load_data(file_dir="./.downloads", download_dir="./.downloads"):
    ''' Function that yields records from the IMDB reviews dataset

        @Arguments:
            file_dir -- personal system file path to the
                unzipped IMDB data set (so, a directory). If this does
                not exist, the archive will be downloaded and unzipped here
            download_dir -- what directory to download the actual archive to? Can be None,
                in which case it defaults to the parent directory of file_path.
                The archive will only be downloaded if necessary

        @Return:
            A generator over a tuples of Movie reviews and their sentiment
    '''
    # Open file path
    imdb_root = os.path.join(file_dir, "aclImdb")
    if not os.path.isdir(imdb_root):
        logger.info("Downloading IMDB dataset")
        if download_dir is None:
            download_dir = os.path.dirname(os.path.normpath(file_dir))
        downloaded_file_path = get_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                             download_dir)
        # then extract it 
        if not os.path.isdir(os.path.join(file_dir, 'aclImdb')):
            logger.info("Extracting IMDB dataset")
            tar = tarfile.open(downloaded_file_path, mode="r:gz")
            tar.extractall(path=file_dir)
            tar.close()

    imdb_train = os.path.join(imdb_root, "train")
    imdb_test = os.path.join(imdb_root, "test")
    imdb_train_pos = os.path.join(imdb_train, "pos")
    imdb_train_neg = os.path.join(imdb_train, "neg")
    imdb_test_pos = os.path.join(imdb_test, "pos")
    imdb_test_neg = os.path.join(imdb_test, "neg")
    
    # Specifies positive and negative files
    pos_train = os.listdir(imdb_train_pos)
    pos_train = [(os.path.join(imdb_train_pos, file_name), pos_label) for file_name in pos_train]
    pos_test = os.listdir(imdb_test_pos)
    pos_test = [(os.path.join(imdb_test_pos, file_name), pos_label) for file_name in pos_test]

    neg_train = os.listdir(imdb_train_neg)
    neg_train = [(os.path.join(imdb_train_neg, file_name), neg_label) for file_name in neg_train]
    neg_test = os.listdir(imdb_test_neg)
    neg_test = [(os.path.join(imdb_test_neg, file_name), neg_label) for file_name in neg_test]

    all_data = pos_train + pos_test + neg_train + neg_test

    # Combines data and shuffles it.
    random.shuffle(all_data)

    for (file_path, sentiment) in all_data:
        # Open the movie review
        f = open(file_path, 'r')
        yield (f.read().decode('utf-8'), sentiment)
        # Closes f on the following next() call by user
        f.close()

def main():
    data = load_data("/root/data/pcallier/imdb",None)
    print data.next()

if __name__=="__main__":
    main()
