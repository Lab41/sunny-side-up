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
from sklearn.linear_model import LogisticRegression
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
    all_data = open_stanford_twitter_csv(PATH)

    # Generator over all_data helps to save memory usage
    all_data = [LabeledSentence(tweet.lower().split(), sent)
                for tweet, sent in iter(all_data)]

    # Builds model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, 
                    workers=7)
    model.build_vocab(all_data)

    for epoch in range(10):
        model.train(shuffle(all_data))

    if save_model:
        model.save(model_name)

    return


def usage():
    # TODO
    print("TODO")
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
