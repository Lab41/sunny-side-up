from __future__ import absolute_import
from __future__ import print_function

import os,sys
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/examples')[0]
sys.path.insert(0, ssu_path)


#print(sys.path)
#x = os.path.dirname(os.path.abspath(__file__))
#print(x)
#print(os.path.dirname(os.path.abspath(__file__)))
#print(x.rsplit('/examples')[0])
#print(x.rsplit('/src'))

import simplejson as json
import pandas as pd
import numpy as np
import pprint
import time
from datasets import imdb
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
from keras.datasets import cifar10, imdb


'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python tufs_cnn.py
'''

def model_defn():
    print('Build model...')
  
    #Set Parameters for model
    fully_connected = [8704,1024,1024,1]
    
    model = Sequential()

    model.add(Convolution2D(256,67,7,input_shape=(1,67,1014),border_mode="valid"))
    #model.add(Convolution1D(256,7,input_dim=1,input_shape=(67,1014),border_mode="valid"))
    #model.add(MaxPooling1D(pool_length=3))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #model.add(Convolution1D(256,7))
    #model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(Convolution2D(256,1,7))
    #model.add(MaxPooling1D(pool_length=3))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #model.add(Convolution1D(256,3))
    model.add(Convolution2D(256,1,3))

    model.add(Convolution2D(256,1,3))

    model.add(Convolution2D(256,1,3))

    model.add(Convolution2D(256,1,3))
    #model.add(MaxPooling1D(pool_length=3))
    model.add(MaxPooling2D(pool_size=(1,3)))

    model.add(Flatten())

    #Fully Connected Layers 
    model.add(Dense(fully_connected[1]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(fully_connected[2]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(fully_connected[3]))
    model.add(Activation('sigmoid'))
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")
    
    return model

if __name__=="__main__":

    # get training and testing sets, and their sizes for amazon.
    # this HDF5 file uses an 80/20 train/test split and lives at /data/pcallier/amazon
    '''(amtr, amte), (amntr, amnte) = datasets, sizes = batch_data.split_data(
        batch_data.batch_data(amazon_reviews.load_data("sampleAmz.json.gz")), 
        h5_path="sampleAmz.hd5", 
        overwrite_previous=True,
        in_memory=False)
    
    (amtr, amte), (amntr, amnte) = datasets, sizes = batch_data.split_data(
        batch_data.batch_data(amazon_reviews.load_data("reviews_Home_and_Kitchen.json.gz")), 
        h5_path="amazon_split.hd5", 
        overwrite_previous=True,
        in_memory=False)
    '''

    (amtr,amte),(amntr,amnte) = datasets,sizes = batch_data.split_data(None,h5_path='amazon_split.hd5',
        overwrite_previous=False,shuffle=True)
    
    am_train_batch = batch_data.batch_data(amtr,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x[0]),
        flatten=False)

    am_test_batch = batch_data.batch_data(amte,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x[0]),
        flatten=False)

    #reviews,labels = am_train_batch.next()
    #print("shape of stuffs",reviews.shape)

    '''# batch training, testing sets
    am_train_batch = batch_data.batch_data(amtr,
        normalizer_fun=None,transformer_fun=None)
    am_test_batch = batch_data.batch_data(amte,
        normalizer_fun=None,transformer_fun=None)
    '''
    reviews,labels = am_train_batch.next()
    
    #print(reviews[0].sum(axis=1))
    print(type(reviews))
    print("shape of stuffs",reviews.shape)
    trans = reviews[:,np.newaxis]
    print("Transformed shape", trans.shape)
    #trans = numpy.array(map(lambda x:numpy.array((x,x*2,x*3)), my_ar))
    #print("sizes of train", amntr)
    #print("sizes of test", amnte)
    

    num_epochs = 1
    model = model_defn()
    count = 0

    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        progbar = generic_utils.Progbar(amntr)

        
        for X_batch, Y_batch in am_train_batch:
            #if (count > 10):
            #    break
            X_batch = X_batch[:,np.newaxis]
            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(128, values=[("train loss", loss),("train acc",acc)])
            #model.save_weights('testSave.hd5',overwrite=True)
            count = count + 1

        print("\nTesting...")

        progbar = generic_utils.Progbar(amnte)
        count = 0
        for X_batch, Y_batch in am_test_batch:
            #if (count > 10):
            #    break
            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(128, values=[("test loss", loss),("test acc",acc)])
            count = count + 1   

        print("\n")