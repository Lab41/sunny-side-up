import os
import tarfile
import random
from data_utils import get_file


# Dictionary that defines the Sentiment features
#Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}
pos_label = 1
neg_label = 0

def load_data(file_path=None, download_path="./.downloads", dest_path="./.downloads"):
    ''' Function that takes in a path to the IMDB movie review dataset
        word analogy file, opens it, removes topic tags and returns a list
        of the analogies

        @Arguments:
            file_path -- (optional) personal system file path to the
                IMDB data set in gzip form(or others of
                a similar structure)

        @Return:
            A generator over a tuples of Movie reviews and their sentiment
    '''
    # Open file path
    if not file_path:
        print "Downloading IMDB dataset"
        file_path = get_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                             download_path)

    # If file has not been extracted, then extract it 
    # to the downloads folder. This will save a lot of time
    if not os.path.isdir(os.path.join(dest_path, 'aclImdb')):
        print("Extracting IMDB dataset")
        tar = tarfile.open(file_path, mode="r:gz")
        tar.extractall(path=dest_path)
        tar.close()

    # Specifies positive and negative files
    pos_train = os.listdir('./.downloads/aclImdb/train/pos')
    pos_train = [(os.path.join('./.downloads/aclImdb/train/pos', file_name), pos_label) for file_name in pos_train]
    pos_test = os.listdir('./.downloads/aclImdb/test/pos')
    pos_test = [(os.path.join('./.downloads/aclImdb/test/pos', file_name), pos_label) for file_name in pos_test]

    neg_train = os.listdir('./.downloads/aclImdb/train/neg')
    neg_train = [(os.path.join('./.downloads/aclImdb/train/neg', file_name), neg_label) for file_name in neg_train]
    neg_test = os.listdir('./.downloads/aclImdb/test/neg')
    neg_test = [(os.path.join('./.downloads/aclImdb/test/neg', file_name), neg_label) for file_name in neg_test]

    all_data = pos_train + pos_test + neg_train + neg_test

    # Combines data and shuffles it.
    random.shuffle(all_data)

    for (file_path, sentiment) in all_data:
        # Open the movie review
        f = open(file_path, 'r')
        yield (f.read(), sentiment)
        # Closes f on the following next() call by user
        f.close()
