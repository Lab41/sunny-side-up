#!/usr/bin/env python
"""
Demo character-level CNN on Neon
"""

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_path))

import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import neon
import neon.layers
import neon.initializers
import neon.transforms
import neon.models
import neon.optimizers
import neon.callbacks
from neon.callbacks.callbacks import Callbacks, Callback
from neon.backends import gen_backend

import src.datasets.amazon_reviews as amazon
import src.datasets.data_utils as data_utils
from src.datasets.batch_data import batch_data, split_data
from src.datasets.data_utils import from_one_hot
from src.datasets.neon_iterator import DiskDataIterator

def get_data(batch_size, doclength=300):
    # batch data and split it to HDF5
    # the generators obtained here are not 
    # used except for debugging
    amz_data = batch_data(amazon.load_data("/root/data/amazon/reviews_Home_and_Kitchen.json.gz"),
        batch_size=batch_size,
        normalizer_fun=lambda x: data_utils.normalize(x, max_length=doclength),
        transformer_fun=None)
    h5_repo = "/root/data/pcallier/amazon/home_kitch_split.hd5"
    (a, b), (a_size, b_size) = split_data(amz_data, 
                      in_memory=False, 
                      h5_path=h5_repo,
                      overwrite_previous=False)
    logger.debug("Test iteration (shape of one record): {}".format(iter(a).next()[0].shape))
    logger.debug("Test iteration (one record): {}".format(iter(a).next()[0]))

    # define functions for use with neon_iterator
    # TODO: create proper iterator classes and use those
    # these functions are returend, alongside the batch sizes
    def a_batcher():
        (a,b),(a_size,b_size)=split_data(None, 
            h5_path=h5_repo, 
            overwrite_previous=False, 
            shuffle=True)
        return batch_data(a, 
            normalizer_fun=lambda x: x,
            transformer_fun=lambda x: data_utils.to_one_hot(x[0]), 
            flatten=True,
            batch_size=batch_size)
    def b_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_repo, overwrite_previous=False)
        return batch_data(b, 
            normalizer_fun=lambda x: x, 
            transformer_fun=lambda x: data_utils.to_one_hot(x[0]), 
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

def crepe_model(nvocab=67, nframes=256, batch_norm=True):
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

class MyCallback(Callback):
    def __init__(self, epoch_freq=1, minibatch_freq=1):
        super(MyCallback, self).__init__(epoch_freq, minibatch_freq)

    def on_train_begin(self, epochs):
        logger.debug("Beginning training ({})".format(epochs))
        logger.debug(dir(self))

    def on_epoch_begin(self, epoch):
        logger.debug("Beginning epoch {}".format(epoch))

    def on_epoch_end(self, epoch):
        logger.debug("Ending epoch {}".format(epoch))

    def on_minibatch_begin(self, epoch, minibatch):
        logger.debug("Epoch {}/Minibatch {} starting".format(epoch,minibatch))
    
    def on_minibatch_end(self, epoch, minibatch):
        logger.debug("Epoch {}/Minibatch {} done".format(epoch,minibatch))

def main():
    batch_size=64
    doc_length=1014
    vocab_size=67
    gpu_id=0
    nframes=256

    model_state_path="/root/data/pcallier/amazon/neon_crepe_model.hd5"
    logger.info("Getting backend...")
    be = gen_backend(backend='gpu', batch_size=batch_size, device_id=0)
    logger.info("Getting data...")
    (train_get, test_get), (train_size, test_size) = get_data(batch_size, doc_length)
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
    callbacks = Callbacks(mlp,train_iter,valid_set=test_iter,valid_freq=3,progress_bar=True)
    callbacks.add_save_best_state_callback(model_state_path)
    #callbacks.add_validation_callback(test_iter, 2)
    optimizer = neon.optimizers.GradientDescentMomentum(
        learning_rate=0.01,
        momentum_coef=0.9,
        schedule=neon.optimizers.Schedule([2,5,8], 0.5))
    #optimizer = neon.optimizers.Adadelta()
    # my_callback = MyCallback()
    # callbacks.add_callback(my_callback, 0)
    logger.info("Doing training...")
    mlp.fit(train_iter, optimizer=optimizer, 
              num_epochs=10, cost=cost, callbacks=callbacks)
    logger.info("Misclassification error: {}".format(mlp.eval(test_iter, metric=neon.transforms.Misclassification())))

if __name__=="__main__":
    main()


