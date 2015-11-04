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

import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def get_imdb(batch_size, doclength, 
             data_path,
             h5_path):
    imdb_data = imdb.load_data(data_path, None)
    imdb_batches = batch_data(imdb_data, batch_size,
        normalizer_fun=lambda x: data_utils.normalize(x, max_length=doclength),
        transformer_fun=None)
    (_, _), (train_size, test_size) = split_data(imdb_batches, h5_path, overwrite_previous=False)
    def train_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_path, overwrite_previous=False, shuffle=True)
        return batch_data(a,
            normalizer_fun=lambda x: x,
            transformer_fun=lambda x: data_utils.to_one_hot(x),
            flatten=True,
            batch_size=batch_size)
    def test_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path, overwrite_previous=False,shuffle=False)
        return batch_data(b,
            normalizer_fun=lambda x: x,
            transformer_fun=lambda x: data_utils.to_one_hot(x),
            flatten=True,
            batch_size=batch_size)

    return (train_batcher, test_batcher), (train_size, test_size)

def get_amazon(batch_size, doclength, data_path, h5_path):
    # batch data and split it to HDF5
    # the generators obtained here are not 
    # used except for debugging
    amz_data = batch_data(amazon.load_data(data_path),
        batch_size=batch_size,
        normalizer_fun=lambda x: data_utils.normalize(x, max_length=doclength),
        transformer_fun=None)
    (a, b), (a_size, b_size) = split_data(amz_data, 
                      h5_path=h5_path,
                      overwrite_previous=False)
    logger.debug("Test iteration (shape of one record): {}".format(iter(a).next()[0].shape))
    logger.debug("Test iteration (one record): {}".format(iter(a).next()[0]))

    # define functions for use with neon_iterator
    # TODO: create proper iterator classes and use those
    # these functions are returned, alongside the batch sizes
    def a_batcher():
        (a,b),(a_size,b_size)=split_data(None, 
            h5_path=h5_path, 
            overwrite_previous=False, 
            shuffle=True)
        return batch_data(a, 
            normalizer_fun=lambda x: x,
            transformer_fun=lambda x: data_utils.to_one_hot(x), 
            flatten=True,
            batch_size=batch_size)
    def b_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_path, overwrite_previous=False)
        return batch_data(b, 
            normalizer_fun=lambda x: x, 
            transformer_fun=lambda x: data_utils.to_one_hot(x), 
            flatten=True,
            batch_size=batch_size)

    return (a_batcher, b_batcher), (a_size, b_size)

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

def crepe_model(nvocab=67, nframes=256, batch_norm=False):
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
    return layers

def do_model(dataset_name, base_dir, data_filename, hdf5_name):
    batch_size=128
    doc_length=1014
    vocab_size=67
    gpu_id=1
    nframes=256
    dataset_loaders = { 'amazon'    : amazon.load_data,
                        'imdb'      : imdb.load_data,
                        'sentiment140'  : sentiment140.load_data }

    present_time = datetime.datetime.strftime(datetime.datetime.now(),"%m%d_%I%p")
    model_state_path=os.path.join(base_dir, "neon_crepe_model_{}.pkl".format(present_time))
    model_weights_history_path=os.path.join(base_dir, "neon_crepe_weights_{}.pkl".format(present_time))
    logger.info("Getting backend...")
    be = gen_backend(backend='gpu', batch_size=batch_size, device_id=gpu_id)
    logger.info("Getting data...")

    data_path = os.path.join(base_dir, data_filename)
    hdf5_path = os.path.join(base_dir, hdf5_name)
    data_loader = dataset_loaders[dataset_name](data_path)
    (train_get, test_get), (train_size, test_size) = split_and_batch(
        data_loader,
        batch_size, 
        doc_length,
        hdf5_path,
        rng_seed=888,
        normalizer_fun=lambda x: data_utils.normalize(x, max_length=doc_length))
    train_batch_beta = test_get()
    logger.debug("First record shape: {}".format(train_batch_beta.next()[0].shape))

    train_iter = DiskDataIterator(train_get, ndata=train_size, 
                                  doclength=doc_length, 
                                  nvocab=vocab_size)
    test_iter = DiskDataIterator(test_get, ndata=test_size, 
                                  doclength=doc_length, 
                                  nvocab=vocab_size)
    logger.info("Building model...")
    model_layers = crepe_model(nframes=nframes)
    mlp = neon.models.Model(model_layers)

    cost = neon.layers.GeneralizedCost(neon.transforms.CrossEntropyBinary())
    callbacks = NeonCallbacks(mlp,train_iter,valid_set=test_iter,valid_freq=3,progress_bar=True)
    callbacks.add_neon_callback(metrics_path=os.path.join(base_dir, "metrics.json"), insert_pos=0)
    callbacks.add_save_best_state_callback(model_state_path)
    callbacks.add_serialize_callback(1, model_weights_history_path,history=5)

    optimizer = neon.optimizers.GradientDescentMomentum(
        learning_rate=0.01,
        momentum_coef=0.9,
        schedule=neon.optimizers.Schedule([2,5,8], 0.5))

    logger.info("Doing training...")
    mlp.fit(train_iter, optimizer=optimizer, 
              num_epochs=10, cost=cost, callbacks=callbacks)
    logger.info("Testing accuracy: {}".format(mlp.eval(test_iter, metric=Accuracy())))

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="Name of dataset (one of amazon, imdb, sentiment140)")
    args = arg_parser.parse_args()
    #do_model(get_amazon, 
    #    base_dir="/root/data/pcallier/amazon/", 
    #    data_filename="reviews_Health_and_Personal_Care.json.gz",
    #    hdf5_name="home_kitch_split.hd5")
    # do_model(get_imdb,
    #     base_dir="/root/data/pcallier/imdb",
    #     data_filename="",
    #     hdf5_name="imdb_split.hd5")
    model_args = { "imdb": {
            'base_dir'      : "/root/data/pcallier/imdb",
            'data_filename' : "",
            'hdf5_name'     : "imdb_split.hd5"},
        'amazon': {
            'base_dir'      : "/root/data/pcallier/amazon",
            'data_filename' : "reviews_Health_and_Personal_Care.json.gz",
            'hdf5_name'     : "home_kitch_split.hd5"            
            },
        'sentiment140': {
            'base_dir'      : "/root/data/pcallier/sentiment140",
            'data_filename' : "sentiment140.csv",
            'hdf5_name'     : "sentiment140_split.hd5"
            }
        }

    dataset_name = args.dataset
    do_model(dataset_name, **model_args[dataset_name])
if __name__=="__main__":
    main()


