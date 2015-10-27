#!/usr/bin/env python

import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import numpy as np
import h5py

import data_utils


def batch_data(data_loader, batch_size=128, normalizer_fun=data_utils.normalize, 
               transformer_fun=data_utils.to_one_hot, flatten=True):
    '''
    Batches data, doing all necessary preprocessing and
    normalization.
    
    If you are batching from HDF5 files, there are probably 
    much faster ways to do this.

    @Arguments:
        data_loader -- an iterable object that yields a single 
            record at a time, as a tuple (data, label).
            Records must be valid input for normalizer_fun;
            by default, this implies string (bytes, unicode)
            data.
        
        batch_size -- how many records should the batcher 
            yield at a time?

        normalizer_fun -- function which takes the first member of the tuple yielded
            by data_loader as its input and returns a normalized
            version of it. The default implements some normalizations
            from Zhang and LeCun's character-level convolutional 
            networks paper.

        transformer_fun -- transforms the output of normalizer_fun into a numpy array.
            Can be used to do one-hot encoding, embedding lookups, etc.
            Output can be any 2+-dimensional numpy array.

        flatten -- should all dimensions of the output of transformer_fun
            be flattened (collapsed to one dimension)? If true, the 
            batch yielded will be 2-D (num batches, record size).

    @Returns:
        generator that yields 2-tuples of (data, label), where data
        and label are numpy arrays representing a batch of data,  
        equally sized in the first dimension
    '''

    docs = []
    labels = []

    logger.debug(data_loader)

    # set (near) identity functions for transformation functions set to None
    if transformer_fun is None:
        transformer_fun = lambda x: np.array(x)
    if normalizer_fun is None:
        logger.debug("Default normalization")
        normalizer_fun = lambda x: x

    # loop over data, applying transforming fns,
    # accumulating records into batches,
    # and yielding them when batch size is reached
    for doc_text, label in data_loader:
        # transformation and normalization
        try:
            logger.debug("Normalization........")
            doc_text = normalizer_fun(doc_text)
            # transform document into a numpy array
            transformed_doc = transformer_fun(doc_text)
            docs.append(transformed_doc)
            labels.append(label)
        except data_utils.DataException as e:
            logger.info(e)

        # dispatch once batch is of appropriate size 
        if len(docs) >= batch_size:
            docs_np = np.array(docs)
            if flatten==True:
                # transform to form (batch_size, w*h); flattening doc
                docs_np = docs_np.reshape((batch_size,-1))
            # labels come out in a separate (batch_size, 1) np array
            labels_np = np.array(labels).reshape((batch_size, -1))
            docs = []
            labels = []

            yield docs_np, labels_np

def pick_splits(splits):
    ''' Pick a bin from a list of n-1 probabilities (0-1)
    for landing in n bins. Used for splitting data.'''
    # avoid changing splits argument in place
    splits = splits[:]
    assert np.sum(splits) < 1.0
    splits.append(1 - np.sum(splits))
    # generate the strictly increasing cutoffs for each split from >0 to 1 
    cum_cutoffs = np.cumsum(splits)
    logger.debug(cum_cutoffs)
    # roll the die
    pick_num = np.random.random()
    logger.debug(pick_num)
    bin_num = 0
    for cutoff in cum_cutoffs:
        if pick_num < cutoff:
            return bin_num
        bin_num += 1
    
class H5Iterator:
    """Small utility class for iterating over an HDF5 file.
    Iterating over it yields tuples of (data, label) from datasets in the
    file with the names given in data_name and labels_name.
    By default, will randomly access records in any given iteration
    """    
    def __init__(self, h5_path, data_name, labels_name, shuffle=True):
        self.h5file = h5py.File(h5_path, "r")
        self.shuffle = shuffle
        self.data = self.h5file[data_name]
        self.labels = self.h5file[labels_name]

        self.indices = range(self.data.shape[0])
        
    def __del__(self):
        self.h5file.close()
    
    def __iter__(self):
        if self.shuffle == True:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        for which_index in indices:
            yield (self.data[which_index], self.labels[which_index])

    #def next(self):
    #    return (self.data.next(), self.labels.next())
        
def write_batch_to_h5(splits, h5_file, data_sizes, new_data, new_labels):
    """ Takes some information about a minibatch of data and 
        writes it into the given HDF5 file

        @Arguments
            splits -- akin to split data, a series of probabilities
                that sum to less than one. The nth probability is the 
                probability of landing in the nth bin. 1-sum(splits) is 
                the probability of landing in the last bin
            h5_file -- an h5py File object, representing an HDF5
                file open for writing
            data_sizes -- a list of length len(splits) + 1 for
                each bin of data. Values are the counts of data in each bin
            new_data -- a numpy array with a new minibatch
            new_labels -- a numpy array with a new set of labels
    """
    # check that data and labels are the same size
    assert new_data.shape[0] == new_labels.shape[0]
    # make a copy of data_sizes
    data_sizes = data_sizes[:]
    # pick which bin to assign data to
    bin_id = pick_splits(splits)
    bin_name = str(bin_id)
    # get slice indexes
    start_i = data_sizes[bin_id]
    end_i = start_i + new_data.shape[0]
    # resize HDF5 datasets
    h5_file["data_" + bin_name].resize(end_i, 0)
    h5_file["labels_" + bin_name].resize(end_i, 0)
    # write data
    h5_file["data_" + bin_name][start_i:end_i, ...] = new_data
    h5_file["labels_" + bin_name][start_i:end_i, ...] = new_labels
    # create and return updated dictionary of bin counts
    data_sizes[bin_id] = end_i
    return data_sizes

def split_data(batch_iterator,
               splits = [0.8],
               rng_seed=None,
               in_memory=False,
               h5_path='/data/amazon/data.hd5',
               overwrite_previous=False,
               shuffle=False):
    ''' Splits data into slices and returns a list of
        iterators over each slice. Slice size is configurable.
        Probabilistic, so may not produce exactly the expected bin sizes, 
        especially for small data.
    
        @Arguments 
            batch_iterator --
                generator of tuples (data, label) where each of data, label
                is a numpy array with the first dimension representing batch size.
                This can be none if in_memory is False, h5_path is valid, and
                overwrite_previous=False (uses existing data, does not re-shuffle 
                or rearrange).
            splits --
                list of floats indicating how to split the data. The data will
                be split into len(splits) + 1 slices, with the final slice 
                having 1-sum(splits) of the data.
            rng_seed -- random number generator seed
            in_memory --
                load data into memory (True) or use HDF5?
            h5_path -- path to HDF5 file. Only used if in_memory is False
            overwrite_previous -- if h5_path is already a readable file,
                overwrite it?
            shuffle -- should the iterators return records in shuffled order?
            
        @Returns
            A 2-tuple:
            The first element is a  list of generators, where each generator
            represents a slice of the data and yields 2-tuples
            of (data,label), each representing one record.
            The second element is an integer-indexed iterable
            with the counts of records in each bin
    '''

    # How many chunks to split into?
    nb_slices = len(splits) + 1
    np.random.seed(rng_seed)
    bin_sizes = [0]*nb_slices
    if in_memory:
        data_bins = None
        for data, labels in batch_iterator:
            bin_i = pick_splits(splits)
            if data_bins == None:
                data_bins = [ (np.ndarray(((0,) + data.shape[1:])), 
                              np.ndarray(((0,) + labels.shape[1:]))) 
                              for a in range(nb_slices) ]
            # store batch in the proper bin, creating numpy arrays
            # for data and labels if needed
            accumulated_data = data_bins[bin_i][0]
            accumulated_labels = data_bins[bin_i][1]
            data_bins[bin_i] = (np.concatenate(
                                (accumulated_data, data)),
                               np.concatenate(
                                (accumulated_labels, labels)))
            # update running tally of data count per bin
            bin_sizes[bin_i] += data.shape[0]
        # return iterators over each bin rather than a list of tuples of np arrays
        # (used to be optional)
        for bin_i in range(nb_slices):
            bin_data, bin_labels = data_bins[bin_i]
            data_bins[bin_i] = iter(zip(bin_data, bin_labels))
        return data_bins, bin_sizes
                    
    else:
        # Check for HDF5 file already on disk
        if overwrite_previous or not os.path.isfile(h5_path):
            with h5py.File(h5_path, "w") as  h5_file:
            
                # get one batch to diagnose dimensions and dtypes
                first_data, first_labels = batch_iterator.next()

                # create one dataset for each slice
                bin_names = [str(bin_i) for bin_i in range(nb_slices)]
                for bin_name in bin_names:
                    h5_file.create_dataset(name="data_" + bin_name,
                                       shape=first_data.shape,
                                       maxshape=(None,) +  first_data.shape[1:],
                                       dtype=first_data.dtype)
                    h5_file.create_dataset(name="labels_" + bin_name,
                                       shape=first_labels.shape,
                                       maxshape=(None,) +  first_labels.shape[1:],
                                       dtype=first_labels.dtype)
                # loop batches into dataset
                # write first batch in
                #data_sizes = {}
                bin_sizes = write_batch_to_h5(splits,h5_file,bin_sizes,first_data,first_labels)
                # then do rest
                for new_data, new_labels in batch_iterator:
                    bin_sizes = write_batch_to_h5(splits, h5_file, bin_sizes, new_data, new_labels)
        else:
            # fill in counts of each data slice
            with h5py.File(h5_path, "r") as f:
                for bin_i in range(nb_slices):
                    try:
                        bin_sizes[bin_i] = f['data_' + str(bin_i)].shape[0]
                    except KeyError:
                        pass
                
        # now to return iterators over the HDF5 datasets for each slice
        # these can, in turn, be batched with batch_data (auughhh)
        data_iterators = []
        for bin_i in range(nb_slices):
            data_iterators.append((H5Iterator(h5_path, 
                "data_" + str(bin_i), 
                "labels_" + str(bin_i),
                shuffle=shuffle)))
        return data_iterators, bin_sizes
        
               
                
if __name__=="__main__":
    # some demo code
    import imdb
    import amazon_reviews
    import batch_data
    import data_utils
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('h5_path', help="Path to pre-split HDF5 file", 
                        default="/data/pcallier/amazon/amazon_split.hd5",
                        nargs='?')
    args = parser.parse_args()

    # get training and testing sets, and their sizes for amazon.
    # this HDF5 file uses an 80/20 train/test split and lives at /data/pcallier/amazon
    (amtr, amte), (amntr, amnte) = datasets, sizes = batch_data.split_data(
        None, 
        h5_path=args.h5_path, 
        overwrite_previous=False,
        in_memory=False)
    import sys

    # batch training, testing sets
    am_train_batch = batch_data.batch_data(amtr,
        normalizer_fun=lambda x: data_utils.normalize(x[0], 
            max_length=300, 
            truncate_left=True),
        transformer_fun=None)
    am_test_batch = batch_data.batch_data(amte,
        normalizer_fun=None,transformer_fun=None)
    
    # Spit out some sample data
    next_batch = am_train_batch.next()
    data, label = next_batch
    np.set_printoptions(threshold=np.nan)
    print "Batch properties:"
    print "Length: {}".format(len(data))
    print "Type: {}".format(type(data))
    print
    print "First record of first batch:"
    print "Type (1 level in): {}".format(type(data[0]))
    print "Type of record (2 levels in): {}".format(type(data[0,0]))
    print data[0,0]
    print "Sentiment label: {}".format(label[0])
    print "In numpy format:"
    oh = data_utils.to_one_hot(data[0,0])
    print np.array_str(np.argmax(oh,axis=0))
    print "Translated back into characters:\n"
    print data_utils.from_one_hot(oh)
    
    # dimension checks
    second_batch_data, second_batch_label = second_batch = am_train_batch.next()
    second_batch = list(second_batch)
    print len(second_batch)
    print "Data object type: ", type(second_batch_data)
    print second_batch_data.shape
    
