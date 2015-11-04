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
This file defines a DataIterator subclass, compatible for use with Nervana Systems'
neon, that accepts a generator function whose value yields minibatches of arbitrary size
and copies them to the neon backend as necessary for use in 
training neon models.
"""
import numpy as np
import gzip

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from neon import NervanaObject

from amazon_reviews import load_data
from batch_data import batch_data as batcher
from data_utils import from_one_hot

class DiskDataIterator(NervanaObject):

    def __init__(self, batch_gen_fun, ndata, doclength, nvocab, nlabels=2):
        """
        Implements loading of given data into backend tensor objects. If the
        backend is specific to an accelarator device, the data is copied over
        to that device.

        Args:
            batch_gen_fun (function): function that returns an iterator over 
                batches (tuples of numpy arrays)
            ndata (int): The number of records in the given data
            nlabels (int, optional): The number of possible types of labels. 2 for binary good/bad
            nvocab  (int, optional): The number of letter tokens
                (not necessary if not providing labels)
        """
        
        # set function for loading data and intialize batch generator
        # this would have been better implemented as an iterator
        self.batch_gen_fun = batch_gen_fun
        self.ndata = ndata
        self.reset()

        self.nlabels = nlabels
        self.nvocab = nvocab
        self.nsteps = doclength

        # on device tensor for review chars and one hot
        self.xlabels_flat = self.be.iobuf((1, self.nsteps), dtype=np.int16)
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
        For resetting the starting index of this dataset back to zero. Used on 
        initialization and every call to __iter__()
        """
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
            # TODO: implement wraparound. as of now, partial batches are discarded
            X, y = next(self.datagen)
            logger.debug("\nX labels shape: {}, X' shape: {}"
                        "\nXbuf shape: {}, Xbuf flat shape: {}"
                        "\nY labels shape: {}, Y buffer shape: {}, Y input shape: {}".format(
                   self.xlabels.shape,
                   X.T.shape,
                   self.Xbuf.shape,
                   self.Xbuf_flat.shape,
                   self.ylabels.shape,
                   self.ybuf.shape,
                   y.shape))
            # This is where data is copied from host memory
            # to the backend. For some reason, it is
            # best to transpose it. X is expected to 
            # already be in the form we want (one-hot
            # or embedded, probably)
            # y is expected to come as a count-from-zero integer,
            # which we convert to one-hot encoding here
            self.ylabels[:] = y.T.copy()
            ylabels = self.ylabels.get()
            logger.debug(ylabels.T.shape)
            zerosum=np.sum([1 for _ in np.nditer(ylabels) if _ == 0])
            onesum=np.sum([1 for _ in np.nditer(ylabels) if _ == 1])
            othersum=np.sum([1 for _ in np.nditer(ylabels) if _ not in (0, 1)])
            #logger.debug("Num 0: {}, Num 1: {}, Other: {}".format(zerosum, onesum, othersum))
            self.Xbuf[:] = X.T.copy()
            self.ybuf[:] = self.be.onehot(self.ylabels, axis=0)
            logger.debug(y.shape)
            #logger.debug(np.concatenate((y, ylabels.T, self.ybuf.get().T), axis=1))
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
