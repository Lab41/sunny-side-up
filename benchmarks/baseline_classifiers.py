import os, sys, logging
import json
import numpy as np
import random
from collections import defaultdict, Counter
import cPickle as pickle

import cProfile, pstats
import threading
import time
import multiprocessing
import math

from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.datasets import data_utils
from src.datasets.data_utils import timed, TextTooShortException, DataSampler, WordVectorBuilder
from src.datasets.imdb import IMDB
from src.datasets.sentiment140 import Sentiment140
from src.datasets.amazon_reviews import AmazonReviews
from src.datasets.open_weiboscope import OpenWeibo
from src.datasets.word_vector_embedder import WordVectorEmbedder

data_fraction_test = 0.20
data_fraction_train = 0.80

num_threads = multiprocessing.cpu_count()
threadLock = threading.Lock()
class dataProcessingThread(threading.Thread):
    '''
        process text,sentiment pair in a thread
        significantly speeds up text vectorization phase
    '''
    def __init__(self, threadID, data, args, values, labels):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.data = data
        self.args = args

    def run(self):

        # process valid entries
        if self.data[0]:

            # process valid data
            text, sentiment = self.data
            if text:
                try:
                    # normalize and tokenize if necessary
                    if self.args.has_key('normalize'):
                        text_normalized = data_utils.normalize(text, **self.args['normalize'])
                    else:
                        text_normalized = text

                    # tokenize
                    tokens = data_utils.tokenize(text_normalized)

                    # choose embedding type
                    vector = None
                    if self.args['embed']['type'] == 'concatenated':
                        vector = embedder.embed_words_into_vectors_concatenated(tokens, **self.args['embed'])
                    elif self.args['embed']['type'] == 'averaged':
                        vector = embedder.embed_words_into_vectors_averaged(tokens)
                    else:
                        pass

                    # data labeled by sentiment score (thread-safe with lock)
                    if vector is not None:
                        threadLock.acquire()
                        values.append(vector)
                        labels.append(sentiment)
                        threadLock.release()

                except TextTooShortException as e:
                    pass





# setup logging
logger = data_utils.syslogger(__name__)

# set output directory
dir_data = "/data"
try:
    dir_results = os.path.join(dir_data, os.path.dirname(os.path.realpath(__file__)), 'results')
except NameError:
    dir_results = os.path.join(dir_data, 'results')

# data inputs
datasets =  {
                'sentiment140': {
                                    'class':    Sentiment140,
                                    'path':     os.path.join(dir_data, 'sentiment140.csv'),
                                    'args':     { 'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'min_length': 70,
                                                                    'max_length': 150,
                                                                    'reverse': False
                                                                },
                                                  'shuffle_after_load': False
                                                }
                                },
                'imdb':         {
                                    'class':    IMDB,
                                    'path':     os.path.join(dir_data, 'imdb'),
                                    'args':     { 'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'encoding': None,
                                                                    'reverse': False
                                                                },
                                                  'shuffle_after_load': False
                                                }
                                },
                'amazon':       {
                                    'class':    AmazonReviews,
                                    'path':     os.path.join(dir_data, 'amazonreviews.gz'),
                                    'args':     { 'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'encoding': None,
                                                                    'reverse': False,
                                                                    'min_length': 0,
                                                                    'max_length': 9999999
                                                                },
                                                  'shuffle_after_load': True
                                                }
                                },
                'openweibo':    {
                                    'class':    OpenWeibo,
                                    'path':     os.path.join(dir_data, 'openweibo'),
                                    'args':     { 'embed':      {   'type': 'averaged' },
                                                  'shuffle_after_load': True,
                                                  'models': {
                                                        'word2vec': {
                                                            'prebuilt_model_path': '/data/openweibo.bin'
                                                        }
                                                  }
                                                }
                                }
            }




# word embeddings
def embedders():
    return ['glove','word2vec']

def classifiers():
    """
        Returns a list of classifier tuples (name, model)
        for use in training
    """
    return [("LogisticRegression", LogisticRegression(C=1.0,
                                                      class_weight=None,
                                                      dual=False,
                                                      fit_intercept=True,
                                                      intercept_scaling=1,
                                                      penalty='l2',
                                                      random_state=None,
                                                      tol=0.0001)),

           ("RandomForests", RandomForestClassifier(n_jobs=-1,
                                                    n_estimators = 15,
                                                    max_features = 'sqrt')),
           ("Gaussian NaiveBayes", GaussianNB()),
           ("LinearSVM", svm.LinearSVC())]




# profiled methods
@timed
def timed_training(classifier, values, labels):
    return classifier.fit(values, labels)

@timed
def timed_testing(classifier, values):
    return classifier.predict(values)


@timed
def timed_dataload(data, args, values, labels):

    # use separate counter to account for invalid input along the way
    counter = 0
    print_every = math.floor(10000/num_threads)*num_threads

    # iterate data
    data_iterator = iter(data)

    # continue processing until no data is left
    value_last = 'initialize'
    while value_last is not None:

        if (counter % print_every == 0):
            print("Loading data at {}...".format(counter))

        # reset subset and threads
        subset = []
        threads = []

        # retrieve the next num_threads entries
        for i in xrange(num_threads):

            # retrieve next entry if available
            pair = next(data_iterator, None)
            if pair is None:
                text = sentiment = None
            else:
                text, sentiment = pair
                counter += 1

            # set the last value for stopping condition
            value_last = text

            # store subset of values for parallel processing
            subset.append( (text, sentiment) )


        # setup threads
        for i in xrange(num_threads):

            # process data in new thread
            thread = dataProcessingThread(i, subset[i], args, values, labels)

            # start thread
            thread.start()

            # add thread to list
            threads.append(thread)


        # wit for all threads to complete
        for t in threads:
            t.join()



# test all vector models
for embedder_model in embedders():

    # iterate all datasources
    for data_source, data_params in datasets.iteritems():

        # prepare data loader
        klass = data_params['class']
        loader = klass(data_params['path'])
        data_args = data_params['args']
        data = loader.load_data()

        # initialize lists (will be converted later into numpy arrays)
        values = []
        labels = []

        # initialize vector embedder
        prebuilt_model_path = data_args.get('models', {}).get(embedder_model, {}).get('prebuilt_model_path', None)
        embedder = WordVectorEmbedder(embedder_model, prebuilt_model_path)

        # load pre-sampled data from disk
        if prebuilt_model_path:
            with open(WordVectorBuilder.filename_train(prebuilt_model_path), 'rb') as f:
                data = pickle.load(f)
        else:

            # get equal-sized subsets of each class
            min_samples = data_args['min_samples'] if data_args.has_key('min_samples') else None
            data_sampler = DataSampler(klass, file_path=data_params['path'], num_classes=2)
            data = data_sampler.sample_balanced(min_samples)

        # load dataset
        logger.info("processing {} samples from {}...".format(len(data), data_params['path']))
        profile_results = timed_dataload(data, data_args, values, labels)

        # store loading time
        seconds_loading = profile_results.timer.total_tt

        # shuffle if necessary
        if data_args['shuffle_after_load']:
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            values = [values[i] for i in indices]
            labels = [labels[i] for i in indices]

        # convert into nparray for sklearn
        values = np.array(values, dtype="float32")
        labels = np.array(labels, dtype="float32")
        logger.info("Loaded {} samples...".format(len(values)))

        # split into training and test data
        logger.info("splitting dataset into training and testing sets...")
        labels_train, labels_dev, labels_test = data_utils.split_data(labels, train=data_fraction_train, dev=0, test=data_fraction_test)
        values_train, values_dev, values_test = data_utils.split_data(values, train=data_fraction_train, dev=0, test=data_fraction_test)
        logger.info("Training on {}, Testing on {}...".format(len(values_train), len(values_test)))


        # calculate distribution
        dist = Counter()
        dist.update(labels_test)


        # setup classifier
        for classifier_name,classifier in classifiers():

            # profiled training
            logger.info("Training %s classifier..." % classifier.__class__.__name__)
            profile_results = timed_training(classifier, values_train, labels_train)
            seconds_training = profile_results.timer.total_tt

            # profiled testing
            logger.info("Testing %s classifier..." % classifier.__class__.__name__)
            profile_results = timed_testing(classifier, values_test)
            predictions = profile_results.results
            seconds_testing = profile_results.timer.total_tt

            # calculate metrics
            data_size           = len(labels_test)
            data_positive       = np.sum(labels_test)
            data_negative       = data_size - data_positive
            confusion_matrix    = metrics.confusion_matrix(labels_test, predictions)
            TN                  = confusion_matrix[0][0]
            FP                  = confusion_matrix[0][1]
            FN                  = confusion_matrix[1][0]
            TP                  = confusion_matrix[1][1]
            accuracy            = metrics.accuracy_score(labels_test, predictions)
            precision           = metrics.precision_score(labels_test, predictions)
            recall              = metrics.recall_score(labels_test, predictions)
            f1                  = metrics.f1_score(labels_test, predictions)

            # build results object
            results = { 'classifier':   str(classifier.__class__.__name__),
                        'data':    {    'source':                   str(data_source),
                                        'testsize':                 str(data_size),
                                        'positive':                 str(data_positive),
                                        'negative':                 str(data_negative),
                                        'time_in_seconds_loading':  str(seconds_loading)
                                   },
                        'embedding': {  'model':                    str(embedder_model),
                                        'subset':                   str(embedder.model_subset)
                                    },
                        'data_args':    data_args,
                        'metrics': {    'TP':                       str(TP),
                                        'FP':                       str(FP),
                                        'TN':                       str(TN),
                                        'FN':                       str(FN),
                                        'accuracy':                 str(accuracy),
                                        'precision':                str(precision),
                                        'recall':                   str(recall),
                                        'f1':                       str(f1),
                                        'time_in_seconds_training': str(seconds_training),
                                        'time_in_seconds_testing':  str(seconds_testing)
                                    }
                       }

            # ensure output directory exists
            if not os.path.isdir(dir_results):
                data_utils.mkdir_p(dir_results)

            # save json file
            filename_results = "{}_{}_{}.json".format(data_source, embedder_model, classifier.__class__.__name__)
            logger.info("Saving results to {}...".format(filename_results))
            with open(os.path.join(dir_results,filename_results), 'a') as outfile:
                json.dump(results, outfile, sort_keys=True, indent=4, separators=(',', ': '))
                outfile.write('\n')
