import os, sys, logging
import json
import numpy as np
from collections import defaultdict, Counter

import cProfile, pstats

from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.datasets import data_utils
from src.datasets.data_utils import timed, TextTooShortException
from src.datasets.imdb import IMDB
from src.datasets.sentiment140 import Sentiment140
from src.datasets.word_vector_embedder import WordVectorEmbedder

data_fraction_test = 0.20
data_fraction_train = 0.80

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
                'imdb':         {
                                    'class':    IMDB,
                                    'path':     os.path.join(dir_data, 'imdb'),
                                    'args':     { 'normalize': {    'encoding': None }

                                                }
                                },
                'sentiment140': {
                                    'class':    Sentiment140,
                                    'path':     os.path.join(dir_data, 'sentiment140.csv'),
                                    'args':     { 'normalize': {    'min_length': 70,
                                                                    'max_length': 150
                                                                }
                                                }
                                }
            }



# word embeddings
def embedders():
    return ['glove']

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
           ("LinearSVM", svm.LinearSVC()),
           ("SVM", svm.SVC())]




# profiled methods
@timed
def timed_training(classifier, values, labels):
    return classifier.fit(values, labels)

@timed
def timed_testing(classifier, values):
    return classifier.predict(values)

@timed
def timed_dataload(loader, data, args, values, labels):
    for i, (text, sentiment) in enumerate(data):

        if (i % int(len(values)/20) == 0):
            logger.info("Embedding {}...".format(i))

        try:
            # input data in the form of paragraph vectors from normalized text
            text_normalized = data_utils.normalize(text, **args['normalize'])
            values[i] = embedder.embed_words_into_vectors_averaged(text_normalized)

            # data labeled by sentiment score
            labels[i] = sentiment

        except TextTooShortException as e:

            # adjust number of valid samples
            loader.samples -= 1
            pass

    # adjust array size
    values = values[:loader.num_samples()]
    labels = labels[:loader.num_samples()]




# test all vector models
for embedder_model in embedders():

    # initialize vector embedder
    embedder = WordVectorEmbedder(embedder_model)

    # iterate all datasources
    for data_source, data_params in datasets.iteritems():

        # prepare data loader
        klass = data_params['class']
        loader = klass(data_params['path'])
        data_args = data_params['args']
        data = loader.load_data()


        # load dataset
        logger.info("loading dataset {}...".format(data_params['path']))
        values = np.zeros((loader.num_samples(), embedder.num_features()), dtype="float32")
        labels = np.zeros(loader.num_samples(), dtype='float32')
        profile_results = timed_dataload(loader, data, data_args, values, labels)
        seconds_loading = profile_results.timer.total_tt


        # split into training and test data
        logger.info("splitting dataset into training and testing sets...")
        labels_train, labels_dev, labels_test = data_utils.split_data(labels, train=data_fraction_train, dev=0, test=data_fraction_test)
        values_train, values_dev, values_test = data_utils.split_data(values, train=data_fraction_train, dev=0, test=data_fraction_test)


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
                        'data':    {    'source': str(data_source),
                                        'testsize': str(data_size),
                                        'positive': str(data_positive),
                                        'negative': str(data_negative),
                                        'time_in_seconds_loading': str(seconds_loading)
                                   },
                        'metrics': {    'TP': str(TP),
                                        'FP': str(FP),
                                        'TN': str(TN),
                                        'FN': str(FN),
                                        'accuracy': str(accuracy),
                                        'precision': str(precision),
                                        'recall': str(recall),
                                        'f1': str(f1),
                                        'time_in_seconds_training': str(seconds_training),
                                        'time_in_seconds_testing': str(seconds_testing)
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
