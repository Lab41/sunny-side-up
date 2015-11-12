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
logging.basicConfig(level=logging.DEBUG)
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
import src.datasets.data_utils as data_utils
from src.datasets.batch_data import batch_data, split_data, split_and_batch
from src.datasets.data_utils import from_one_hot
from src.datasets.neon_iterator import DiskDataIterator
from src.neon.neon_utils import ConfusionMatrixBinary, NeonCallbacks, NeonCallback, Accuracy
from src.datasets.word_vector_embedder import WordVectorEmbedder


def lstm_model(nvocab=67, hidden_size=20, embedding_dim=60):
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

def crepe_model(nvocab=67, nframes=256, batch_norm=False, variant=None):
    init_gaussian = neon.initializers.Gaussian(0, 0.05)
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

        neon.layers.Affine(2,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Logistic())
        
    ]
    # remove certain layers if we are asked for a variant
    if variant == 'embedding' or variant == 'tweet_character':
        del layers[8]
        del layers[3]
    elif variant != None:
        raise Exception("variant must be one of embedding|tweet_character")

    if variant == 'embedding':
        del layers[1] 

    logger.debug(variant)
    logger.debug(layers)
    return layers

def do_model(dataset_name, working_dir, results_path, data_path, hdf5_path, 
        valid_freq=2,
        batch_size=128,
        vocab_size=67,
        learning_rate=0.01,
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
        crepe_variant=None,
        warm_start_path=None,
        **kwargs):
    dataset_loaders = { 'amazon'    : amazon.load_data,
                        'imdb'      : imdb.load_data,
                        'sentiment140'  : sentiment140.load_data }
    if sequence_length==None:
        sequence_length = max_length

    # set up result paths
    present_time = datetime.datetime.strftime(datetime.datetime.now(),"%m%d_%I%p")
    model_state_path=os.path.join(results_path, "neon_crepe_model_{}.pkl".format(present_time))
    model_weights_history_path=os.path.join(results_path, "neon_crepe_weights_{}.pkl".format(present_time))
    metrics_path_template=os.path.join(results_path, "metrics.json")
    logger.debug("\n"
        "Working directory: {}\n"
        "Results path: {}\n"
        "Metrics path: {}\n".format(working_dir, results_path, metrics_path_template))

    logger.info("Getting backend...")
    be = gen_backend(backend='gpu', batch_size=batch_size, device_id=gpu_id, rng_seed=rng_seed)
    logger.info("Getting data...")

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

    data_loader = dataset_loaders[dataset_name](data_path)
    logger.debug("Keyword args: {}".format(kwargs))
    (train_get, test_get), (train_size, test_size) = split_and_batch(
        data_loader,
        batch_size, 
        max_length,
        hdf5_path,
        rng_seed=rng_seed,
        normalizer_fun=normalizer_fun,
        transformer_fun=transformer_fun)
    train_batch_beta = test_get()
    logger.debug("First record shape: {}".format(train_batch_beta.next()[0].shape))

    train_iter = DiskDataIterator(train_get, ndata=train_size, 
                                  doclength=sequence_length, 
                                  nvocab=vocab_size)
    test_iter = DiskDataIterator(test_get, ndata=test_size, 
                                  doclength=sequence_length, 
                                  nvocab=vocab_size)
    logger.info("Building model...")
    model_layers = crepe_model(nvocab=vocab_size,nframes=nframes,variant=crepe_variant)
    mlp = neon.models.Model(model_layers)

    if warm_start_path:
        mlp.load_weights(warm_start_path)

    layers_description = [l.get_description() for l in mlp.layers]
    logger.info(pprint.pformat(layers_description))
    cost = neon.layers.GeneralizedCost(neon.transforms.CrossEntropyBinary())
    callbacks = NeonCallbacks(mlp,train_iter,valid_set=test_iter,valid_freq=valid_freq,progress_bar=True)
    callbacks.add_neon_callback(metrics_path=metrics_path_template, insert_pos=0)
    callbacks.add_save_best_state_callback(model_state_path)
    callbacks.add_serialize_callback(save_freq, model_weights_history_path,history=nr_to_save)

    optimizer = neon.optimizers.GradientDescentMomentum(
        learning_rate=learning_rate,
        momentum_coef=0.9,
        schedule=neon.optimizers.Schedule([2,5,8], 0.5))

    logger.info("Doing training...")
    mlp.fit(train_iter, optimizer=optimizer, 
              num_epochs=nr_epochs, cost=cost, callbacks=callbacks)
    logger.info("Testing accuracy: {}".format(mlp.eval(test_iter, metric=Accuracy())))

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
            }
        }

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="Name of dataset (one of amazon, imdb, sentiment140)")
    arg_parser.add_argument("--working_dir", "-w", default=".",
	    help="Directory where data should be put, default PWD")
    arg_parser.add_argument("--glove", "-g", action="store_true")
    arg_parser.add_argument("--results_dir", "-r", default=None, help="(optional) custom subfolder to store results and weights in (defaults to dataset)")
    arg_parser.add_argument("--data_path", "-d", default=None, help="(optional) custom path to original data")
    arg_parser.add_argument("--hdf5_path", "-5", default=None, help="(optional) custom path to split data in HDF5")
    arg_parser.add_argument("--weights_path", default=None, help="(optional) path to weights to initialize model with")

    args = arg_parser.parse_args()
    dataset_name = args.dataset
    args.working_dir = os.path.abspath(args.working_dir)
    if not args.results_dir:
        args.results_dir = dataset_name
    args.results_dir = os.path.join(args.working_dir, args.results_dir)
    if not args.data_path:
        args.data_path = os.path.join(args.working_dir, model_defaults[dataset_name]['data_filename'])
    if not args.hdf5_path:
        args.hdf5_path = os.path.join(args.working_dir, model_defaults[dataset_name]['hdf5_name'])

    model_args = { 'sentiment140' : {
            'max_length'    : 150,
            'min_length'    : 70,
            'learning_rate' : 1e-4,
            'normalizer_fun': lambda x: data_utils.normalize(x, min_length=70, max_length=150),
            'transformer_fun': data_utils.to_one_hot,
            'variant'       : 'tweet_character',
            },
            'imdb'      : {},
            'amazon'    : {}
        }
    if args.glove:
        glove_embedder = WordVectorEmbedder("glove", os.path.join(args.working_dir, "glove.twitter.27B.zip"))
        model_args[dataset_name]['normalizer_fun'] = lambda x: x.encode('ascii', 'ignore').lower()
        model_args[dataset_name]['transformer_fun'] = lambda x: glove_embedder.embed_words_into_vectors(x, 50)
        model_args[dataset_name]['vocab_size'] = 25
        model_args[dataset_name]['sequence_length'] = 50
        model_args[dataset_name]['crepe_variant'] = 'embedding'

    try:
        logger.debug(model_args[dataset_name]['normalizer_fun'])
    except KeyError:
        logger.debug("No custom normalization fn specified")
    do_model(dataset_name, 
             args.working_dir,
             args.results_dir,
             args.data_path,
             args.hdf5_path,
             **model_args[dataset_name])
if __name__=="__main__":
    main()


