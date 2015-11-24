#!/usr/bin/env python
"""
Demo character-level CNN on Neon
"""

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
two_up = os.path.dirname(os.path.dirname(current_path))
sys.path.append(two_up)
import datetime
import argparse
import pprint
import logging
import json
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

import numpy as np


import neon
import neon.layers
import neon.initializers
import neon.transforms
import neon.models
import neon.optimizers
import neon.callbacks
from neon.callbacks.callbacks import Callbacks, Callback
from neon.backends import gen_backend

import src.datasets.imdb as imdb
import src.datasets.amazon_reviews as amazon
import src.datasets.sentiment140 as sentiment140
import src.datasets.open_weiboscope as open_weiboscope
import src.datasets.data_utils as data_utils
from src.datasets.batch_data import batch_data, split_data, split_and_batch
from src.datasets.data_utils import from_one_hot
from src.datasets.neon_iterator import DiskDataIterator
from src.neon.neon_utils import ConfusionMatrixBinary, NeonCallbacks, NeonCallback, Accuracy
from src.datasets.word_vector_embedder import WordVectorEmbedder
from src.datasets.model_downloader import ModelDownloader

# turn down certain verbose logging levels
#logging.getLogger("src.datasets.batch_data").setLevel(logging.INFO)
#logging.getLogger("src.datasets.neon_iterator").setLevel(logging.INFO)
#logging.getLogger("src.neon.neon_utils").setLevel(logging.INFO)

def lstm_model_draft(nvocab=67, hidden_size=20, embedding_dim=60):
    init_emb = neon.initializers.Uniform(low=-0.1/embedding_dim, high=0.1/embedding_dim)
    layers = [
        neon.layers.LookupTable(vocab_size=nvocab, embedding_dim=embedding_dim, init=init_emb),
        neon.layers.LSTM(hidden_size, neon.initializers.GlorotUniform(),
                         activation=neon.transforms.Tanh(), 
                         gate_activation=neon.transforms.Logistic(),
                         reset_cells=True),
        neon.layers.RecurrentSum(),
        neon.layers.Dropout(0.5),
        neon.layers.Affine(2, neon.initializers.GlorotUniform(),
                           bias=neon.initializers.GlorotUniform(),
                           activation=neon.transforms.Softmax())
        ]
    return layers

def lstm_model(hidden_size=200, noutput=2):
    layers = [
        neon.layers.LSTM(hidden_size,
                         init=neon.initializers.GlorotUniform(),
                         bias=neon.initializers.Constant(1),
                         activation=neon.transforms.Tanh(),
                         gate_activation=neon.transforms.Logistic()),
        neon.layers.Dropout(0.5),
        neon.layers.Affine(noutput, 
                           init=neon.initializers.GlorotUniform(),
                           bias=neon.initializers.GlorotUniform(),
                           activation=neon.transforms.Logistic())
        ]
    return layers

def simple_model(nvocab=67,
                 doclength=1014):
    layers = [
        neon.layers.Conv((10,10,100),
            init=neon.initializers.Gaussian(),
            activation=neon.transforms.Rectlin()),

        neon.layers.Pooling((10,10)),

       # neon.layers.Pooling((3, 3)),

        neon.layers.Affine(nout=256,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),

        neon.layers.Affine(nout=2,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin())
    ]
    return layers

def crepe_model(nvocab=67, nframes=256, gaussian_sd=0.05,
                batch_norm=False, noutput=2, variant=None):
    init_gaussian = neon.initializers.Gaussian(0, gaussian_sd)
    layers = [
        neon.layers.Conv((nvocab, 7, nframes), 
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling((1, 3)),

        neon.layers.Conv((1, 7, nframes), 
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling((1, 3)),

        neon.layers.Conv((1, 3, nframes),
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),
        
        neon.layers.Conv((1, 3, nframes),
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),
            
        neon.layers.Conv((1, 3, nframes), 
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),

        neon.layers.Conv((1, 3, nframes), 
            batch_norm=batch_norm,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling((1, 3)),

        neon.layers.Affine(1024,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),

        neon.layers.Dropout(0.5),

        neon.layers.Affine(1024,
            init=init_gaussian,
            activation=neon.transforms.Rectlin()),

        neon.layers.Dropout(0.5),

        neon.layers.Affine(noutput,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Logistic())
        
    ]
    # remove certain layers if we are asked for a variant
    if variant in ('embedding50', 'embedding99') or variant == 'tweet_character':
        del layers[8]       # last pooling
        del layers[3]       # 2nd pooling
    elif variant != None:
        raise Exception("variant must be one of embedding|tweet_character")

    if variant in ('embedding50', ):
        del layers[1]       # 1st pooling

    logger.debug(variant)
    logger.debug(layers)
    return layers

def do_model(dataset_name, working_dir, results_path, data_path, hdf5_path, 
        model_type='cnn',
        valid_freq=2,
        batch_size=256,
        vocab_size=67,
        learning_rate=0.01,
        momentum_coef=0.9,
        min_length=100,
        max_length=1014,
        sequence_length=None,
        rng_seed=888,
        normalizer_fun=data_utils.normalize,
        transformer_fun=data_utils.to_one_hot,
        gpu_id=1,
        nframes=256,
        nr_epochs=30,
        nr_to_save=5,
        save_freq=2,
        hidden_size=200,
        crepe_variant=None,
        warm_start_path=None,
        max_records=None,
        balance_labels=False,
        **kwargs):
    dataset_loaders = { 'amazon'            : amazon.load_data,
                        'imdb'              : imdb.load_data,
                        'sentiment140'      : sentiment140.load_data,
                        'open_weiboscope'   : open_weiboscope.load_data }
    if sequence_length==None:
        sequence_length = max_length

    # set up result paths
    present_time = datetime.datetime.strftime(datetime.datetime.now(),"%m%d_%I%p")
    model_state_path=os.path.join(results_path, "neon_crepe_model_{}.pkl".format(present_time))
    model_weights_history_path=os.path.join(results_path, "neon_crepe_weights_{}.pkl".format(present_time))
    metrics_path_template=os.path.join(results_path, "metrics.json")
    metadata_path=os.path.join(results_path, "metadata.json")
    logger.debug("\n"
        "Working directory: {}\n"
        "Results path: {}\n"
        "Metrics path: {}\n".format(working_dir, results_path, metrics_path_template))

    # record run metadata
    run_metadata = {
        'dataset_name'      : dataset_name,
        'batch_size'        : batch_size,
        'nr_epochs'         : nr_epochs,
        'learning_rate'     : learning_rate,
        'momentum_coef'     : momentum_coef,
        'model_type'        : model_type,
        'model_variant'     : crepe_variant,
        'rng_seed'          : rng_seed,
    }
    if model_type=='lstm':
        run_metadata['hidden_size'] = hidden_size
    elif model_type=='cnn':
        run_metadata['model_variant'] = crepe_variant
    try:
        os.mkdir(working_dir)
    except OSError:
        logger.debug("Was trying to create working directory, but it may already exist",
            exc_info=True)
    try:
        os.mkdir(results_path)
    except OSError:
        logger.debug("Was trying to create results directory, but it may already exist",
            exc_info=True)
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    logger.info("Getting backend...")
    be = gen_backend(backend='gpu', batch_size=batch_size, device_id=gpu_id, rng_seed=rng_seed)
    logger.info("Getting data...")

    # Get helper functions to load dataset
    data_loader = dataset_loaders[dataset_name](data_path)
    (train_get, test_get), (train_size, test_size) = split_and_batch(
        data_loader,
        batch_size, 
        max_length,
        hdf5_path,
        rng_seed=rng_seed,
        normalizer_fun=normalizer_fun,
        transformer_fun=transformer_fun,
        balance_labels=balance_labels,
        max_records=max_records)
    # diagnostic peek at the data
    train_batch_beta = test_get()
    logger.debug("First record shape: {}".format(train_batch_beta.next()[0].shape))

    # Create neon DataIterators for train and test sets
    train_iter = DiskDataIterator(train_get, ndata=train_size, 
                                  doclength=sequence_length, 
                                  nvocab=vocab_size,
                                  nlabels=2,
                                  labels_onehot=True)
    test_iter = DiskDataIterator(test_get, ndata=test_size, 
                                  doclength=sequence_length, 
                                  nvocab=vocab_size,
                                  nlabels=2,
                                  labels_onehot=True)

    logger.info("Building model...")
    # get model layers for either convnet or lstm
    if model_type=='cnn':
        model_layers = crepe_model(nvocab=vocab_size,nframes=nframes,variant=crepe_variant,batch_norm=True)
    elif model_type=='lstm':
        model_layers = lstm_model(hidden_size=hidden_size)

    mlp = neon.models.Model(model_layers)

    # start from saved weights
    if warm_start_path:
        mlp.load_weights(warm_start_path)

    # Set cost
    cost = neon.layers.GeneralizedCost(neon.transforms.CrossEntropyBinary())
    #cost = neon.layers.GeneralizedCost(neon.transforms.SumSquared())
    #cost = neon.layers.GeneralizedCost(neon.transforms.CrossEntropyMulti())
    
    # create callbacks for saving model weights and diagnostic statistics
    callbacks = NeonCallbacks(mlp,train_iter,valid_set=test_iter,valid_freq=valid_freq,progress_bar=True)
    callbacks.add_neon_callback(metrics_path=metrics_path_template, insert_pos=0)
    callbacks.add_save_best_state_callback(model_state_path)
    callbacks.add_serialize_callback(save_freq, model_weights_history_path,history=nr_to_save)

    # learning rate schedule
    decay_schedule = neon.optimizers.Schedule([n for n in range(1, nr_epochs) if n % 3 == 2 and n < 10], 0.5)
    #decay_schedule = neon.optimizers.Schedule()

    # Possible optimizers
    sgd = neon.optimizers.GradientDescentMomentum(
        learning_rate=learning_rate,
        momentum_coef=momentum_coef,
        schedule=decay_schedule)
    rmsprop = neon.optimizers.RMSProp(learning_rate=learning_rate)
    optimizer = sgd

    logger.info("Doing training...")
    # main training loop
    mlp.fit(train_iter, optimizer=optimizer, 
              num_epochs=nr_epochs, cost=cost, callbacks=callbacks)

def normalize_tweet(txt):
    return data_utils.normalize(txt, min_length=70, max_length=150)

def transform_for_vectors(txt):
    txt = data_utils.tokenize(txt[::-1])
    return txt

def normalize_imdb(txt):
    return data_utils.normalize(txt, encoding=None)

def main():
    model_defaults = {
        'imdb': {
            'data_filename' : "",
            'hdf5_name'     : "imdb_split.hd5"},
        'amazon': {
            'data_filename' : "reviews_Health_and_Personal_Care.json.gz",
            'hdf5_name'     : "health_personal_split.hd5"            
            },
        'sentiment140': {
            'data_filename' : "sentiment140.csv",
            'hdf5_name'     : "sentiment140_split.hd5" 
            },
        'open_weiboscope': {
            'data_filename' : "",
            'hdf5_name'     : "open_weiboscope.hd5"
            },
        }

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="Name of dataset (one of amazon, imdb, sentiment140, open_weiboscope)")
    arg_parser.add_argument("--working_dir", "-w", default=".",
	    help="Directory where data and results should be put, default PWD.")
    #arg_parser.add_argument("embedding", choices=('glove','word2vec'), required=False)
    vector_group=arg_parser.add_mutually_exclusive_group()
    vector_group.add_argument("--glove", nargs=1, metavar="LOCATION",help="Use glove, object at this path")
    vector_group.add_argument("--word2vec", nargs=1, metavar="LOCATION",help="Use word2vec, object at this path")
    arg_parser.add_argument("--results_dir", "-r", default=None, help="custom subfolder to store results and weights in (defaults to dataset)")
    arg_parser.add_argument("--data_path", "-d", default=None, help="custom path to original data, partially overrides working_dir")
    model_types=arg_parser.add_mutually_exclusive_group()
    model_types.add_argument("--cnn", default=True, action="store_true", help="Use convolutional model")
    model_types.add_argument("--lstm", action="store_true", help="Use LSTM")
    arg_parser.add_argument("--hdf5_path", "-5", default=None, help="custom path to split data in HDF5")
    arg_parser.add_argument("--weights_path", default=None, help="path to weights to initialize model with")
    arg_parser.add_argument("--gpu_id", "-g", default=0, type=int, help="GPU device ID (integer)")
    arg_parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate, default 0.01")
    arg_parser.add_argument("--momentum_coef", default=0.9, type=float, help="Momentum coefficient, default 0.9")
    arg_parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    arg_parser.add_argument("--nframes", default=256, type=int, help="Frame buffer size for CREPE. 256 or 1024.")
    arg_parser.add_argument("--log_level", default=logging.INFO, type=int)

    args = arg_parser.parse_args()
    logging.getLogger().setLevel(args.log_level)
    dataset_name = args.dataset
    args.working_dir = os.path.abspath(args.working_dir)
    if not args.results_dir:
        args.results_dir = dataset_name
    args.results_dir = os.path.join(args.working_dir, args.results_dir)
    if not args.data_path:
        args.data_path = os.path.join(args.working_dir, model_defaults[dataset_name]['data_filename'])
    if not args.hdf5_path:
        args.hdf5_path = os.path.join(args.working_dir, model_defaults[dataset_name]['hdf5_name'])

    # dataset-specific arguments to do_model
    model_args = { 
        'sentiment140' : {
            'min_length'        : 70,
            'max_length'        : 150,
            'normalizer_fun'    : normalize_tweet,
            'variant'           : 'tweet_character',
            },
        'imdb' : {
            'normalizer_fun'    : normalize_imdb,
            },
        'amazon'            : {
            'normalizer_fun'    : data_utils.normalize, 
            },
        'open_weiboscope'   : {
            'normalizer_fun'    : data_utils.normalize,
            'balance_labels'    : True,
            'max_records'       : 2e6,
            },
        }
    # use 50 words in embedding-based models of microblogs,
    # otherwise use 99 words for other embedding-based models
    if dataset_name in ('sentiment140','open_weiboscope'):
        embedding_nr_words = 50
    else:
        embedding_nr_words = 99

    # CNN or LSTM
    if args.cnn:
        model_args[dataset_name]['model_type'] = 'cnn'
        # character-based by default (overridden for embedding-based, below)
        model_args[dataset_name]['transformer_fun'] = data_utils.to_one_hot
    elif args.lstm:
        model_args[dataset_name]['model_type'] = 'lstm'
        # for character-based LSTM, re-reverse data
        if not (args.glove or args.word2vec):
            model_args[dataset_name]['transformer_fun'] = reverse_one_hot

    # set default hidden size (overridden for embedding-based models below)
    model_args[dataset_name]['hidden_size'] = 10
    model_args[dataset_name]['nframes']=args.nframes
    # parameters for embedding-based models
    if args.glove or args.word2vec:
        model_args[dataset_name]['sequence_length'] = embedding_nr_words
        model_args[dataset_name]['crepe_variant'] = 'embedding{}'.format(embedding_nr_words)
	model_args[dataset_name]['hidden_size'] = 200
    if args.glove:
        glove_path = os.path.abspath(args.glove[0])
        if not os.path.isfile(glove_path):
            model_downloader = ModelDownloader('glove')
            glove_url = model_downloader.data_location['twitter-2b']['url']
            glove_dir = os.path.dirname(glove_path)
            glove_file = os.path.basename(glove_path)
            model_downloader.download_and_save_vectors(glove_url, glove_dir, glove_file)
        glove_embedder = WordVectorEmbedder("glove",glove_path) 
        model_args[dataset_name]['transformer_fun'] = \
            lambda x: glove_embedder.embed_words_into_vectors(
                transform_for_vectors(x), embedding_nr_words)
        model_args[dataset_name]['vocab_size'] = 200
    if args.word2vec:
        w2v_embedder = WordVectorEmbedder("word2vec", os.path.abspath(args.word2vec[0]))
        model_args[dataset_name]['transformer_fun'] = \
            lambda x: w2v_embedder.embed_words_into_vectors(
                transform_for_vectors(x), embedding_nr_words)
        model_args[dataset_name]['vocab_size'] = 300

    try:
        logger.debug(model_args[dataset_name]['normalizer_fun'])
    except KeyError:
        logger.debug("No custom normalization fn specified")
    do_model(dataset_name, 
             args.working_dir,
             args.results_dir,
             args.data_path,
             args.hdf5_path,
             gpu_id=args.gpu_id,
             learning_rate=args.learning_rate,
             momentum_coef=args.momentum_coef,
             batch_size=args.batch_size,
             **model_args[dataset_name])
if __name__=="__main__":
    main()
