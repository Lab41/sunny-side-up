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
                                    'args':     { 'load':       {   'rng_seed': 13337 },
                                                  'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'min_length': 70,
                                                                    'max_length': 150,
                                                                    'reverse': False,
                                                                    'pad_out': False
                                                                },
                                                  'shuffle_after_load': False,
                                                  'models': [
                                                        'glove',
                                                        'word2vec'
                                                  ]
                                                }
                                },
                'imdb':         {
                                    'class':    IMDB,
                                    'path':     os.path.join(dir_data, 'imdb'),
                                    'args':     { 'load':       {   'rng_seed': 13337 },
                                                  'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'encoding': None,
                                                                    'reverse': False,
                                                                    'pad_out': False,
                                                                    'min_length': 0,
                                                                    'max_length': 9999999
                                                                },
                                                  'shuffle_after_load': False,
                                                  'models': [
                                                        'glove',
                                                        'word2vec'
                                                  ]
                                                }
                                },
                'amazon':       {
                                    'class':    AmazonReviews,
                                    'path':     os.path.join(dir_data, 'amazonreviews.gz'),
                                    'args':     { 'load':       {   'rng_seed': 13337 },
                                                  'embed':      {   'type': 'averaged' },
                                                  'normalize':  {   'encoding': None,
                                                                    'reverse': False,
                                                                    'min_length': 0,
                                                                    'max_length': 9999999,
                                                                    'pad_out': False
                                                                },
                                                  'shuffle_after_load': True,
                                                  'models': [
                                                        'glove',
                                                        'word2vec',
                                                        {
                                                            'word2vec':   {   'model': '/data/amazon/amazon_800000.bin' }
                                                        }
                                                  ]
                                                }
                                },
                'openweibo':    {
                                    'class':    OpenWeibo,
                                    'path':     os.path.join(dir_data, 'openweibo'),
                                    'args':     { 'load':       {   'rng_seed': 13337 },
                                                  'embed':      {   'type': 'averaged' },
                                                  'shuffle_after_load': True,
                                                  'models': [
                                                        'glove',
                                                        'word2vec',
                                                        {
                                                            'word2vec':   {   'model': '/data/openweibo/openweibo_800000.bin' }
                                                        }
                                                  ]
                                                }
                                }
,
                'openweibo':    {
                                    'class':    OpenWeibo,
                                    'path':     os.path.join(dir_data, 'openweibocensored'),
                                    'args':     { 'load':   {   'form': 'hanzi',
                                                                'randomseed': 13337
                                                            },
                                                  'embed':      {   'type': 'averaged' },
                                                  'shuffle_after_load': True,
                                                  'models': [
                                                        'glove',
                                                        'word2vec',
                                                        {
                                                            'word2vec':   {   'model': '/data/openweibocensored/openweibo_hanzi.bin' } #TODO
                                                        }
                                                  ]
                                                }
                                }
            }




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

    for text,sentiment in data:

        try:
            if (counter % 10000 == 0):
                print("Loading at {}".format(counter))

            # normalize and tokenize if necessary
            if args.has_key('normalize'):
                text_normalized = data_utils.normalize(text, **args['normalize'])
            else:
                text_normalized = text

            # tokenize
            if data_args.get('load', {}).get('form', None) == 'hanzi':
                tokens = data_utils.tokenize_hanzi(text_normalized)
            else:
                tokens = data_utils.tokenize(text_normalized)

            # choose embedding type
            vector = None
            if args['embed']['type'] == 'concatenated':
                vector = embedder.embed_words_into_vectors_concatenated(tokens, **self.args['embed'])
            elif args['embed']['type'] == 'averaged':
                vector = embedder.embed_words_into_vectors_averaged(tokens)
            else:
                pass

            # data labeled by sentiment score (thread-safe with lock)
            if vector is not None:
                values.append(vector)
                labels.append(sentiment)
                counter += 1

        except TextTooShortException as e:
            pass


# iterate all datasources
for data_source, data_params in datasets.iteritems():

    # prepare data loader
    klass = data_params['class']
    loader = klass(data_params['path'])
    data_args = data_params['args']
    load_args = data_args.get('load', {})
    data = loader.load_data(load_args)

    # test all vector models
    for embedder_model in data_args['models']:

        # identify prebuilt model if exists
        prebuilt_path_model = None
        if isinstance(embedder_model, dict):
            embedder_model, prebuilt_model_params = embedder_model.items().pop()
            prebuilt_path_model = prebuilt_model_params.get('model')

        # initialize word vector embedder
        embedder = WordVectorEmbedder(embedder_model, prebuilt_path_model)

        # load pre-sampled data from disk
        if prebuilt_path_model:

            # training data (custom or default)
            if prebuilt_model_params.get('train', None):
                prebuilt_path_train = prebuilt_model_params.get('train')
            else:
                prebuilt_path_train = WordVectorBuilder.filename_train(prebuilt_path_model)

            # testing data (custom or default)
            if prebuilt_model_params.get('test', None):
                prebuilt_path_test = prebuilt_model_params.get('test')
            else:
                prebuilt_path_test = WordVectorBuilder.filename_test(prebuilt_path_model)

            # import pickled data
            with open(prebuilt_path_train, 'rb') as f:
                data_train = pickle.load(f)
            with open(prebuilt_path_test, 'rb') as f:
                data_test = pickle.load(f)

            # update embedder parameters
            model_path_dir, model_path_filename, model_path_filext = WordVectorBuilder.filename_components(prebuilt_path_model)
            embedder.model_group = model_path_filename
            embedder.model_subset = model_path_filename

            # initialize lists (will be converted later into numpy arrays)
            values_train = []
            labels_train = []
            values_test = []
            labels_test = []

            # initialize timer
            seconds_loading = 0
            logger.info("processing {} samples from {}...".format(len(data_train)+len(data_test), prebuilt_path_model))

            # load training dataset
            profile_results = timed_dataload(data_train, data_args, values_train, labels_train)

            # store loading time
            seconds_loading += profile_results.timer.total_tt

            # load training dataset
            profile_results = timed_dataload(data_test, data_args, values_test, labels_test)

            # store loading time
            seconds_loading += profile_results.timer.total_tt

            # create numpy arrays for classifier input
            values_train = np.array(values_train, dtype='float32')
            labels_train = np.array(labels_train, dtype='float32')
            values_test = np.array(values_test, dtype='float32')
            labels_test = np.array(labels_test, dtype='float32')

            # shuffle if necessary
            if data_args['shuffle_after_load']:
                np.random.shuffle(values_train)
                np.random.shuffle(labels_train)
                np.random.shuffle(values_test)
                np.random.shuffle(labels_test)

        else:

            # initialize lists (will be converted later into numpy arrays)
            values = []
            labels = []

            # get equal-sized subsets of each class
            data_sampler = DataSampler(klass, file_path=data_params['path'], num_classes=2)
            data = data_sampler.sample_balanced(min_samples=data_args.get('min_samples', None), rng_seed=data_args.get('load', {}).get('rng_seed', None))

            # load dataset
            logger.info("processing {} samples from {}...".format(len(data), data_params['path']))
            profile_results = timed_dataload(data, data_args, values, labels)

            # store loading time
            seconds_loading = profile_results.timer.total_tt

            # convert into nparray for sklearn
            values = np.array(values, dtype="float32")
            labels = np.array(labels, dtype="float32")
            logger.info("Loaded {} samples...".format(len(values)))

            # shuffle if necessary
            if data_args['shuffle_after_load']:
                np.random.shuffle(values)
                np.random.shuffle(labels)

            # split into training and test data
            logger.info("splitting dataset into training and testing sets...")
            labels_train, labels_dev, labels_test = data_utils.split_data(labels, train=data_fraction_train, dev=0, test=data_fraction_test)
            values_train, values_dev, values_test = data_utils.split_data(values, train=data_fraction_train, dev=0, test=data_fraction_test)


        # calculate distribution
        dist = Counter()
        dist.update(labels_test)


        # setup classifier
        logger.info("Training on {}, Testing on {}...".format(len(values_train), len(values_test)))
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
