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
    ## NOTE! ##
    # Setting min_count > 1 can cause some tweets to "disappear" later #
    # from the Doc2Vec sentence corpus. #
    # ex: you could imagine a tweet containing only words whose count was low #
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5,
                    workers=1)

    logging.info("Building Vocabulary...")
    model.build_vocab(labeled_sent)

    logging.info("Training model...")
    for epoch in range(epoch_num):
        logging.info("Epoch %s..." % epoch)
        # Temporarily sets logging level to show only if its at least WARNING
        # This prevents model.train from overloading the log
        logging.getLogger().setLevel(logging.WARN)
        model.train(np.random.permutation(labeled_sent))
        logging.getLogger().setLevel(logging.INFO)

    return model


def to_sklearn_format(model):
    # This function is specific to the Sentiment140 Dataset
    train_arrays = np.zeros((72000 * 2, 100))
    train_labels = np.zeros(72000 * 2)
    test_arrays = np.zeros((8000 * 2, 100))
    test_labels = np.zeros(8000 * 2)
    for i in range(72000):
        prefix_train_pos = 'pos_' + str(i)
        prefix_train_neg = 'neg_' + str(i)
        train_arrays[i] = model[prefix_train_pos]
        train_arrays[72000 + i] = model[prefix_train_neg]
        train_labels[i] = 1
        train_labels[72000 + i] = 0
    for i in range(8000):
        prefix_test_pos = 'pos_' + str(i)
        prefix_test_neg = 'neg_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_arrays[8000 + i] = model[prefix_test_neg]
        test_labels[i] = 1
        test_labels[8000 + i] = 0

    return train_arrays, train_labels, test_arrays, test_labels


def test_model(model):
    logging.info("Developing training and testing sets...")
    # Converts data to Sklearn acceptable numpy format
    train_arr, train_labels, test_arr, test_labels = to_sklearn_format(model)

    logging.info("Building logisitic regression classifier...")
    classifier = LogisticRegression()
    classifier.fit(train_arr, train_labels)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, penalty='l2', random_state=None,
                       tol=0.0001)
    logging.info("Accuracy: %.2f" % classifier.score(test_arr, test_labels))


def main(argv):
    try:
        long_flags = ["help", "save", "test", "verbose"]
        opts, args = getopt.getopt(argv, "hi:s:tv", long_flags)
    except:
        usage()
        sys.exit(2)

    model_name = None
    verbose = False
    testing = False
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-s", "--save"):
            model_name = arg
        elif opt in ("-t", "-test"):
            testing = True
        elif opt in ("-v", "--verbose"):
            verbose = True
            output_format = '%(asctime)s : %(levelname)s : %(message)s'
            logging.basicConfig(format=output_format, level=logging.INFO)

    ## TODO ##
    ## Remove ability to remove @mentions
    ## Using a better tokenizer
    logging.info("Opening CSV file...")
    all_data = sentiment140.load_data(verbose=verbose)
    model = train_d2v_model(all_data, epoch_num=10)

    # Saves a ton of memory
    model.init_sims(replace=True)

    if model_name:
        model.save(model_name)

    if testing:
        test_model(model)


def usage():
    print('Usage: Word2Vec_Pipeline.py [-i file | -s file | -h]')
    print('Options and arguments:')
    print('-h\t\t: print this help message and exit (also --help)')
    print('-s model_name\t: saves the model creatred by Doc2Vec (also --help)')
    print('-v\t\t: makes the operation verbose by showing logging (also --verbose)')
    print('')
    print('--help\t\t: print this help message and exit (also -h)\n')
    print('--save\t\t: saves the model creatred by Doc2Vec(also -s)\n')
    print('--test\t\t: runs given test_model function at end of process\n')
    print('--verbose\t\t: makes the operation verbose by showing logging (also -v)')


if __name__ == "__main__":
    main(sys.argv[1:])
