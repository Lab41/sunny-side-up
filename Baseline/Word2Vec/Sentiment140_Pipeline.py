import os, sys
import getopt

# Adds ability to import form datasets
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/Baseline')[0]

sys.path.insert(0, ssu_path)

from datasets import sentiment140
from datasets.data_utils import preprocess_tweet

# Word2Vec/Doc2Vec packages
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression, RandomForestClassifier
from sklearn import metrics


def train_d2v_model(data, epoch_num=10):
    ''' Trains the Doc2Vec on the sentiment140 dataset

    @Arguments:
        data -- the loaded sentiment140 dataset from module

        epoch_num -- sets the number of epochs to train on

    @Return:
        A trained Doc2Vec model
    '''
    labeled_sent = list()
    pos_count = neg_count = 0
    ls = None

    ''' Sets the label for each individual sentence in the Doc2Vec model.
    These become "special words" that allow the vector for a sentence to
    be accessed from the model. Each label must be unique '''
    for (sentence, label) in data:
        if label == 'pos':
            ''' Doc2Vec model takes in only this LabeledSentence data structure
            ex: LabeledSentence(['list', 'of', 'tokenized', 'words'], ['pos_0'])'''
            ls = LabeledSentence(preprocess_tweet(sentence).split(), [label + '_%d' % pos_count])
            pos_count += 1
        else:
            ls = LabeledSentence(preprocess_tweet(sentence).split(), [label + '_%d' % neg_count])
            neg_count += 1
        labeled_sent.append(ls)

    logging.info("Training on %d Positive and %d Negative tweets" % (pos_count, neg_count))
    logging.info("Building model...")

    '''Setting min_count > 1 can cause some tweets to "disappear" later
    from the Doc2Vec sentence corpus.
    ex: you could imagine a tweet containing only words whose count was low'''
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5,
                    workers=7)

    logging.info("Building Vocabulary...")
    model.build_vocab(labeled_sent)

    logging.info("Training model...")
    for epoch in xrange(epoch_num):
        logging.info("Epoch %s..." % epoch)
        # Temporarily sets logging level to show only if its at least WARNING
        # This prevents model.train from overloading the log
        logging.getLogger().setLevel(logging.WARN)
        # Numpy random permutation method shuffles data in place 
        # Shuffling improves the accuracy of the model
        model.train(np.random.permutation(labeled_sent))
        logging.getLogger().setLevel(logging.INFO)

    return model


def to_sklearn_format(model, test=.1):
    ''' Uses a properly trained Doc2Vec model for the
    Sentiment140 dataset and splits the data into training
    and a testing set and put them into numpy array for use
    with scikit learn

    @Arguments:
        model -- A trained and loaded Doc2Vec model of Sentiment140

        test -- the percentage split between training and testing data

    @Raises:
        ValueError -- if test is less than 0 or greater then 1

    @Return:
        4 numpy arrays consisting of the train/testing data and labels
    '''
    if test <= 0 or test >= 1:
        raise ValueError('test variable must be between 0-1')

    test_size = int(80000 * test)
    train_size = 80000 - test_size

    # Initializes numpy data matrices and label vectors
    train_arrays = np.zeros((train_size * 2, 100))
    train_labels = np.zeros(train_size * 2)
    test_arrays = np.zeros((test_size * 2, 100))
    test_labels = np.zeros(test_size * 2)
    for i in xrange(train_size):
        prefix_train_pos = 'pos_' + str(i)
        prefix_train_neg = 'neg_' + str(i)
        ## This relies on previous function ##
        ## Labeling is in blah  blah
        train_arrays[i] = model[prefix_train_pos]
        train_arrays[train_size + i] = model[prefix_train_neg]
        # Positive = 1, Negative = 0
        train_labels[i] = 1
        train_labels[train_size + i] = 0
    for i in xrange(test_size):
        prefix_test_pos = 'pos_' + str(i)
        prefix_test_neg = 'neg_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_arrays[test_size + i] = model[prefix_test_neg]
        # Positive = 1, Negative = 0
        test_labels[i] = 1
        test_labels[test_size + i] = 0

    return train_arrays, train_labels, test_arrays, test_labels


def test_model(model):
    ''' Uses a loaded Doc2Vec model and a LogisticRegression
    from the scikitlearn package to build a sentiment classifier

    @Argument:
        model -- A trained and loaded Doc2Vec model of Sentiment140
    '''
    logging.info("Developing training and testing sets...")
    # Converts data to Sklearn acceptable numpy format
    train_arr, train_labels, test_arr, test_labels = to_sklearn_format(model, test=.1)

    logging.info("Building logisitic regression classifier...")
    #classifier = LogisticRegression(C=1.0, class_weight=None, dual=False,
    #                                fit_intercept=True, intercept_scaling=1,
    #                                penalty='l2', random_state=None, tol=0.0001)
    classifier = RandomForestClassifier(n_estimators = 100)
    classifier.fit(train_arr, train_labels)

    print("Accuracy: %.4f" % classifier.score(test_arr, test_labels))
    print(metrics.classification_report(test_arr, test_labels, target_names=['neg', 'pos']))
    print(metrics.confusion_matrix(test_arr, test_labels))


def main(argv):
    try:
        long_flags = ["help", "save", "test", "verbose"]
        opts, args = getopt.getopt(argv, "hs:tv", long_flags)
    except:
        usage()
        sys.exit(2)

    model_name = None
    testing = False
    verbose = False
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-s", "--save"):
            print('Saving model to %s' % arg)
            model_name = arg
        elif opt in ("-t", "--test"):
            testing = True
        elif opt in ("-v", "--verbose"):
            verbose = True
            output_format = '%(asctime)s : %(levelname)s : %(message)s'
            logging.basicConfig(format=output_format, level=logging.INFO)

    # Prevents user from running script without saving
    # or testing the model
    if not (model_name or testing):
        logging.critical("Sentiment140_Pipeline script is neither saving or testing the model built")
        sys.exit()

    logging.info("Opening CSV file...")
    all_data = sentiment140.load_data(verbose=verbose)
    model = train_d2v_model(all_data, epoch_num=10)

    # Saves memory
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
    print('-t\t\t: runs given test_model function at end of process (also --test)')
    print('-v\t\t: makes the operation verbose by showing logging (also --verbose)')
    print('')
    print('--help\t\t: print this help message and exit (also -h)\n')
    print('--save\t\t: saves the model creatred by Doc2Vec(also -s)\n')
    print('--test\t\t: runs given test_model function at end of process (also -t)\n')
    print('--verbose\t\t: makes the operation verbose by showing logging (also -v)')


if __name__ == "__main__":
    main(sys.argv[1:])
