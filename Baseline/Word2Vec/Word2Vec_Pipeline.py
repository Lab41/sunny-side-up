import sys
import getopt
sys.path.insert(0, '../../datasets')

import sentiment140
from data_utils import split_data

# Word2Vec/Doc2Vec packages
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import logging

# Numpy
import numpy as np

# Sci Kit Learn Classifier
from sklearn.linear_model import LogisticRegression


def train_d2v_model(data, epoch_num=10):
    labeled_sent = list()
    pos_count = 0
    neg_count = 0
    ls = None
    for i, (sentence, label) in enumerate(data):
        if label == 'pos':
            ls = LabeledSentence(sentence.lower().split(), [label + '_%d' % pos_count])
            pos_count += 1
        else:
            ls = LabeledSentence(sentence.lower().split(), [label + '_%d' % neg_count])
            neg_count += 1
        labeled_sent.append(ls)

    logging.info("Training on %d Positive and %d Negative tweets" % (pos_count, neg_count))
    logging.info("Building model...")
    model = Doc2Vec(min_count=3, window=10, size=100, sample=1e-4, negative=5, workers=1)

    logging.info("Building Vocabulary...")
    model.build_vocab(labeled_sent)

    # Sets logging level to show only if its at least WARNING
    # This prevents model.train from overloading the log
    logging.getLogger().setLevel(logging.WARNING)

    logging.info("Training model...")
    ## TODO variable for the number of epochs ##
    for epoch in range(epoch_num):
        logging.info("Epoch %s..." % epoch)
        model.train(np.random.permutation(labeled_sent))

    return model


def to_sklearn_format(data):
    pass


def test_model(model_path):
    logging.info("Loading model...")
    model = Doc2Vec.load(model_path)

    logging.info("Developing training and testing sets...")
    # Converts data to Sklearn acceptable numpy format
    train_arrays = np.zeros((72000 * 2, 100))
    train_labels = np.zeros(72000 * 2)
    test_arrays = np.zeros((8000 * 2, 100))
    test_labels = np.zeros(8000 * 2)
    for i in range(72000 * 2):
        prefix_train_pos = 'pos_' + str(i)
        prefix_train_neg = 'neg_' + str(i)
        train_arrays[i] = model[prefix_train_pos]
        train_arrays[72000 + i] = model[prefix_train_neg]
        train_labels[i] = 1
        train_labels[72000 + i] = 0
    for i in range(8000 * 2):
        prefix_test_pos = 'pos_' + str(i)
        prefix_test_neg = 'neg_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_arrays[8000 + i] = model[prefix_test_neg]
        test_labels[i] = 1
        test_labels[8000 + i] = 0

    logging.info("Building logisitic regression classifier...")
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    logging.info("Accuracy: "),
    classifier.score(test_arrays, test_labels)


def main(argv):

    PATH = './StanfordTweetData/training.1600000.processed.noemoticon.csv'

    try:
        opts, args = getopt.getopt(argv, "hi:s:v", ["help"])
    except:
        usage()
        sys.exit(2)

    save_model = False
    model_name = None
    verbose = False
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

    ## TODO ##
    ## Remove ability to remove @mentions
    ## Using a better tokenizer
    logging.info("Opening CSV file...")
    all_data = sentiment140.load_data(verbose=verbose)
    train_set, dev_set, test_set = split_data(all_data, train=.9, dev=0, test=.1, shuffle=True)
    model = train_d2v_model(all_data, epoch_num=8)

    if save_model:
        model.save(model_name)





def usage():
    print('Usage: Word2Vec_Pipeline.py [-i file | -s file | -h]')
    print('Options and arguments:')
    print('-h\t\t: print this help message and exit (also --help)')
    print('-i csv_file\t: specify local path for StanfordTweet input CSV file')
    print('-s model_name\t: saves the model creatred by Doc2Vec')
    print('-v\t\t: makes the operation verbose')
    print('')
    print('--help\t\t: print this help message and exit (also -h)\n')


if __name__ == "__main__":
    main(sys.argv[1:])
