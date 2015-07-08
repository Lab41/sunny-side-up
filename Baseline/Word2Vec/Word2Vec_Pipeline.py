import sys
import getopt
sys.path.insert(0, '../Bayes-NLTK')

from import_stanford_twitter import open_stanford_twitter_csv

# Word2Vec/Doc2Vec packages
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import logging

# Numpy
import numpy as np

# Sci Kit Learn Classifier
# from sklearn.linear_model import LogisticRegression
from random import shuffle


def main(argv):

    PATH = './StanfordTweetData/training.1600000.processed.noemoticon.csv'

    try:
        opts, args = getopt.getopt(argv, "hi:s:v", ["help"])
    except:
        usage()
        sys.exit(2)

    save_model = False
    model_name = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt == "-i":
            PATH = arg
        elif opt == "-s":
            save_model = True
            model_name = arg
        elif opt == "-v":
            output_format = '%(asctime)s : %(levelname)s : %(message)s'
            logging.basicConfig(format=output_format, level=logging.INFO)

    print("Opening CSV file...")
    all_data = open_stanford_twitter_csv(PATH, verbose=True)

    # Generator over all_data helps to save memory usage
    all_data = [LabeledSentence(tweet.lower().split(), sent) for tweet, sent in all_data]

    # Builds model
    print("Building model...")
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    print("Building Vocabulary...")
    model.build_vocab(all_data)

    # Stops logging, I think that it is slowing things down alot
    logger = logging.getLogger()
    logger.disabled = True

    print("Training model...")
    for epoch in range(10):
        print("Epoch %s..." % epoch)
        shuffle(all_data)
        model.train(all_data)

    if save_model:
        model.save(model_name)


def usage():
    print('Usage: Word2Vec_Pipeline.py [-i file | -s file | -h]')
    print('Options and arguments:')
    print('-h\t\t: print this help message and exit (also --help)')
    print('-i csv_file\t: specify path for StanfordTweet input CSV file')
    print('-s model_name\t: saves the model creatred by Doc2Vec')
    print('-v\t\t: makes the operation verbose')
    print('')
    print('--help\t\t: print this help message and exit (also -h)\n')


if __name__ == "__main__":
    main(sys.argv[1:])
