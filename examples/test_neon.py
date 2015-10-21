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
logger.setLevel(logging.DEBUG)

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
from src.datasets.batch_data import batch_data, split_data
from src.datasets.data_utils import from_one_hot
from src.datasets.neon_iterator import DiskDataIterator

def get_data():
    batch_size=32
    amz_data = batch_data(amazon.load_data("/root/data/amazon/test_amazon.json.gz"),
        batch_size=batch_size)

    h5_repo = "/root/data/pcallier/amazon/demo_split.h5"
    (a, b), (a_size, b_size) = split_data(amz_data, in_memory=False, 
                      h5_path=h5_repo)
    logger.debug("Test iteration: {}".format(a.next()[0].shape))
    def a_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_repo, overwrite_previous=False)
        return batch_data(a, normalizer_fun=lambda x: x,
            transformer_fun=lambda x: x, flatten=False,
            batch_size=batch_size)
    def b_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_repo, overwrite_previous=False)
        return batch_data(b, normalizer_fun=lambda x: x, 
            transformer_fun=lambda x: x, flatten=False,
            batch_size=batch_size)

    return (a_batcher, b_batcher), (a_size, b_size)

def lstm_model(nvocab=67, doclength=1014):
    layers = [
        neon.layers.LSTM(hidden_size, neon.initializers.GlorotUniform(),
                         activation=neon.transforms.Tanh(), 
                         gate_activation=neon.transforms.Logistic(),
                         reset_cells=True),
        neon.layers.RecurrentSum(),
        neon.layers.Dropout(0.5),
        neon.layers.Affine(2, neon.initializers.GlorotUniform())
        ]
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

def get_model(nvocab=67,
          doclength=1014):
    layers = [
        neon.layers.Conv((nvocab, 7, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling(3),

        neon.layers.Conv((1, 7, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling(3),

        neon.layers.Conv((1, 3, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),
        
        neon.layers.Conv((1, 3, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),
            
        neon.layers.Conv((1, 3, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),

        neon.layers.Conv((1, 3, 256), batch_norm=True,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),
        neon.layers.Pooling(3),

        neon.layers.Affine(1024,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),

        neon.layers.Affine(1024,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin()),

        neon.layers.Affine(2,
            init=neon.initializers.Uniform(),
            activation=neon.transforms.Rectlin())
        
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
    batch_size=32
    be = gen_backend(backend='gpu', batch_size=batch_size)
    (train_get, test_get), (train_size, test_size) = get_data()
    train_batch_beta = test_get()
    logger.debug("First record shape: {}".format(train_batch_beta.next()[0].shape))

    train_iter = DiskDataIterator(train_get, ndata=train_size, batch_size=batch_size)
    test_iter = DiskDataIterator(test_get, ndata=test_size, batch_size=batch_size)
    #print ''.join(from_one_hot(a.next()[0][0].reshape((-1,1014))))[::-1]
    model = simple_model()
    mlp = neon.models.Model(model)
    cost = neon.layers.GeneralizedCost(neon.transforms.CrossEntropyBinary())
    callbacks = Callbacks(mlp,train_iter,valid_set=test_iter,valid_freq=3,progress_bar=True)
    my_callback = MyCallback()
    callbacks.add_callback(my_callback, 0)
    logger.debug(callbacks.callbacks)
    mlp.fit(train_iter, optimizer=neon.optimizers.RMSProp(), 
              num_epochs=10, cost=cost, callbacks=callbacks)
    print "Misclassification error: {}".format(model.eval(test_iter, metric=neon.layers.Misclassification()))

if __name__=="__main__":
    main()


