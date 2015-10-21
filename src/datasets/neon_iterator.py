# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
This file defines a DataIterator class, compatible for use with Nervana Systems'
neon, that accepts an iterator over minibatches of arbitrary size
and copies them to the neon backend as necessary for use in 
training neon models.
"""
import numpy as np
#import h5py
#import simplejson as json
import gzip

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from neon import NervanaObject

from amazon_reviews import load_data
from batch_data import batch_data as batcher
from data_utils import from_one_hot

# parameters: data modeling
# path_to_amazon = '/mnt/data/AmazonReviews/aggressive_dedup.json.gz'
# total_records = 82836502
# path_to_amazon = '/mnt/data/AmazonReviews/reviews_Health_and_Personal_Care.json.gz'
# total_records =  2982356

class DiskDataIterator(NervanaObject):

    def __init__(self, batch_gen_fun, ndata, doclength=1014, nvocab=67, nlabels=2):
        """
        Implements loading of given data into backend tensor objects. If the
        backend is specific to an accelarator device, the data is copied over
        to that device.

        Args:
            batch_gen_fun (function): function that returns an iterator over 
                batches (tuples of numpy arrays)
            ndata (int): The number of records in the given data
            nlabels (int, optional): The number of possible types of labels. 2 for binary good/bad
            nvocab  (int, optional): Tne number of letter tokens
                (not necessary if not providing labels)
        """
        # Treat singletons like list so that iteration follows same syntax
        #self.ofile = h5py.File(fname, "r")
        #self.dbuf = self.ofile['reviews']
        
        # set function for loading data and intialize batch generator
        self.batch_gen_fun = batch_gen_fun
        self.ndata = ndata
        self.reset()
        
        #might not work, beware
        #self.be.bsz=batch_size

        self.nlabels = nlabels
        self.nvocab = nvocab
        self.nsteps = doclength  # removing 1 for the label at the front

        # on device tensor for review chars and one hot
        self.xlabels_flat = self.be.iobuf((1, self.nsteps), dtype=np.int32)
        self.xlabels = self.xlabels_flat.reshape((self.nsteps, self.be.bsz))
        self.Xbuf_flat = self.be.iobuf((self.nvocab, self.nsteps))
        self.Xbuf = self.Xbuf_flat.reshape((self.nvocab * self.nsteps, self.be.bsz))

        self.ylabels = self.be.iobuf(1, dtype=np.int32)
        self.ybuf = self.be.iobuf(self.nlabels)

        # This makes upstream layers interpret each example as a 1d image
        self.shape = (1, self.nvocab, self.nsteps)
        self.Xbuf.lshape = self.shape  # for backward compatibility

    @property
    def nbatches(self):
        return self.ndata // self.be.bsz

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        #self.start = 0
        self.datagen = self.batch_gen_fun()

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """
        self.reset()
        for i1 in range(0, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            # anticipate wraparound
            #if i2 == self.ndata:
            #self.start = self.be.bsz - bsz
            # copy labels from HDF5 file/generator
            X, y = next(self.datagen)
            logger.info("Dest shape: {}, Src shape: {}, Xbuf flat shape: {}, Xbuf shape: {}"
                        "\nY labels shape: {}, Y buffer shape: {}, Y input shape: {}".format(
                    self.xlabels.shape,
                    X.T.shape,
                    self.Xbuf_flat.shape,
                    self.Xbuf.shape,
                    self.ylabels.shape,
                    self.ybuf.shape,
                    y.shape))
            #self.xlabels[:] = X.T.copy()
            self.ylabels[:] = y.T.copy()
            # wraparound condition
            #if self.be.bsz > bsz:
            #    self.xlabels[:, bsz:] = self.dbuf[:self.start, 1:].T.copy()
            #    self.ylabels[:, bsz:] = self.dbuf[:self.start, 0:1].T.copy()

            #self.Xbuf_flat[:] = self.be.onehot(self.xlabels_flat, axis=0)
            self.Xbuf[:] = X.T.copy()
            self.ybuf[:] = self.be.onehot(self.ylabels, axis=0)
            yield (self.Xbuf, self.ybuf)


if __name__ == '__main__':
    main()

def main():    
    h5file = '/root/data/pcallier/amazon/temp.hd5'
    amzn_path = '/root/data/pcallier/amazon/reviews_Health_and_Personal_Care.json.gz'
    #azbw = AmazonBatchWriter(amzn_path, h5file)
    #azbw.run()

    from neon.backends.nervanagpu import NervanaGPU
    ng = NervanaGPU(0, device_id=1)

    NervanaObject.be = ng
    ng.bsz = 128
    train_set = DiskDataIterator(lambda: batcher(load_data('/root/data/amazon/test_amazon.json.gz')), 3000, 128, nvocab=67)
    # random examples from each
    for bidx, (X_batch, y_batch) in enumerate(train_set):
        print "Batch {}:".format(bidx)
        #print X_batch.get().T.sum(axis=1)
        reviewnum = input("Pick review index to fetch and decode: ")
        review = from_one_hot(X_batch.get().T[reviewnum].reshape(67, -1))
        print ''.join(review)[::-1]
