import os, sys
import getopt
import logging
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline

# Adds ability to import form datasets
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/Baseline')[0]

sys.path.insert(0, ssu_path)

from datasets import sentiment140
from datasets.data_utils import split_data
from feature_evaluator import test_model
from feature_extractors import word_feats


def main(argv):
    # Initial local path for Stanford Twitter Data Features is None
    FEAT_PATH = None
    verbose = False

    # Parse command line arguments
    try:
        long_flags = ["help", "bernoulli", "multinomial", "gaussian"]
        opts, args = getopt.getopt(argv, "hf:v", long_flags)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # Classifier variable. Used for training on tweet features below
    classifier = NaiveBayesClassifier
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt == '-f':
            FEAT_PATH = arg
        elif opt == '-v':
            verbose = True
            output_format = '%(asctime)s : %(levelname)s : %(message)s'
            logging.basicConfig(format=output_format, level=logging.INFO)
        elif opt in ("bernoulli", "multinomial", "gaussian"):
            ''' This section allows you to use scikit-learn packages for
            text classification.

            NLTKs SklearnClassifier makes the process much easier,
            since you dont have to convert feature dictionaries to
            numpy arrays yourself, or keep track of all known features.
            The Scikits classifiers also tend to be more memory efficient
            than the standard NLTK classifiers, due to their use of sparse
            arrays.

            Credit to "Jacob" and his post on Steamhacker.com
            '''
            pipeline = None
            if opt == "bernoulli":
                pipeline = Pipeline([('nb', BernoulliNB())])
            elif opt == "multinomial":
                pipeline = Pipeline([('nb', MultinomialNB())])
            elif opt == "gaussian":
                pipeline = Pipeline([('nb', GaussianNB())])
            classifier = SklearnClassifier(pipeline)

    # Perform tweet parsing and learning
    logging.info("Opening CSV file...")
    logging.info("Extracting Features...")

    all_data = list()
    # Checks if all_data has already been set
    if FEAT_PATH is not None:
        tweet_feats = open(FEAT_PATH, 'r')
        all_data = [eval(line) for line in tweet_feats]
    else:
        all_data = sentiment140.load_data(feat_extractor=word_feats,
                                          verbose=verbose)

    logging.info("CSV file opened and features extracted")
    train_set, dev_set, test_set = split_data(all_data, train=.9,
                                              dev=0, test=.1, shuffle=True)
    logging.info("Data split into sets")
    classifier = classifier.train(train_set)
    logging.info("Classifier trained")

    logging.info("Evaluating accuracy and other features\n")
    test_model(classifier, test_set)


def usage():
    print('Usage: doall.py [-i file | -h]')
    print('Options and arguments:')
    print('-h\t\t: print this help message and exit (also --help)')
    print('-f txt_file\t: specify path for txt file containing tweet features')
    print('')
    print('--bernoulli\t: specifies the use a Bernoulli Naive Bayes classifier')
    print('\t\t  from the nltk\'s scikit learn integration')
    print('--multinomial\t: specifies the use a Multinomial Naive Bayes')
    print('\t\t  classifier from the nltk\'s scikit learn integration')
    print('--gaussian\t: specifies the use a Gaussian Naive Bayes classifier')
    print('\t\t  from the nltk\'s scikit learn integration')
    print('--help\t\t: print this help message and exit (also -h)\n')

if __name__ == '__main__':
    main(sys.argv[1:])
