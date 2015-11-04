from __future__ import absolute_import
from __future__ import print_function

import os,sys
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/examples')[0]
sys.path.insert(0, ssu_path)

import simplejson as json
import numpy as np
import pprint
import time

from datasets import amazon_reviews
from datasets import batch_data
from datasets import data_utils

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, MaxPooling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils,generic_utils


'''
Model below is based on paper by Xiang Zhang "Character-Level Convolutional
Networks for Text Classification" (http://arxiv.org/abs/1509.01626) paper was
formerly known as "Text Understanding from Scratch" (http://arxiv.org/pdf/1502.01710v4.pdf) 

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python tufs_cnn.py
'''

def model_defn():

    print('Build model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet x 1014
    model.add(Convolution2D(256,67,7,input_shape=(1,67,1014)))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 336 x 256
    model.add(Convolution2D(256,1,7))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 110 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 108 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 106 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 104 X 256
    model.add(Convolution2D(256,1,3))
    model.add(MaxPooling2D(pool_size=(1,3)))

    model.add(Flatten())

    #Fully Connected Layers

    #Input is 8704 Output is 1024 
    model.add(Dense(fully_connected[0]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1024
    model.add(Dense(fully_connected[1]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1
    model.add(Dense(fully_connected[2]))
    model.add(Activation('sigmoid'))
    
    #Stochastic gradient parameters as set by paper
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")
    
    return model,sgd

if __name__=="__main__":

    #Set batch size for input data
    batch_size = 128

    #Set the number of epochs to run
    num_epochs = 10

    #Import model 
    model,sgd = model_defn()

    # Get training and testing sets, and their sizes for the amazon dataset
    # from HDF5 file that uses an 80/20 train/test split 
    (amtr,amte),(amntr,amnte) = datasets, sizes = batch_data.split_data(None,
        h5_path='amazon_split.hd5', overwrite_previous=False,shuffle=True)
    
    #Generator that outputs Amazon training data in batches with specificed parameters
    am_train_batch = batch_data.batch_data(amtr,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x),
        flatten=False, batch_size=batch_size)

    #Generator that outputs Amazon testing data in batches with specificed parameters
    am_test_batch = batch_data.batch_data(amte,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x),
        flatten=False, batch_size=batch_size)

    #Begin runs of training and testing    
    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        #Halve the learning rate every 3 epochs 
        if(e % 3 == 2):
            sgd.lr.set_value(sgd.lr.get_value()/2)
        
        progbar = generic_utils.Progbar(amntr)

        for X_batch, Y_batch in am_train_batch:

            #Reshape input from a 3D Input to a 4D input for training    
            X_batch = X_batch[:,np.newaxis]
            
            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("train loss", loss),("train acc",acc)])
        
        #Save the model every epoch into hd5 file    
        model.save_weights('tufs_keras_weights.hd5',overwrite=True) 

        print("\nTesting...")

        progbar = generic_utils.Progbar(amnte)

        for X_batch, Y_batch in am_test_batch:
            
            #Reshape input from a 3D Input to a 4D input for training
            X_batch = X_batch[:,np.newaxis]

            loss,acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("test loss", loss),("test acc",acc)])
             
        print("\n")

