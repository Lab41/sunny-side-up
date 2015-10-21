#!/usr/bin/env python

import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    logger.debug(dir(data_loader))

    # set (near) identity functions for transformation functions set to None
    if transformer_fun is None:
        transformer_fun = lambda x: np.array(x)
    if normalizer_fun is None:
        normalizer_fun = lambda x: x

    for doc_text, label in data_loader:
        try:
            doc_text = normalizer_fun(doc_text)
            # transform document into a numpy array
            transformed_doc = transformer_fun(doc_text)
            docs.append(transformed_doc)
            labels.append(label)
        except data_utils.DataException as e:
            logger.info(e)
            continue
            
        
        if len(docs) >= batch_size:
            logger.debug(type(docs))
            for doc in docs:
                logger.debug(doc.shape)
            logger.debug("")
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
    Yields tuples of (data, label) from datasets in the
    file with the names given in data_name and labels_name.
    """    
    # TODO: implement resetting
    def __init__(self, h5_path, data_name, labels_name):
        self.h5file = h5py.File(h5_path, "r")
        self.data = iter(self.h5file[data_name])
        self.labels = iter(self.h5file[labels_name])
        
    def __del__(self):
        self.h5file.close()
    
    def __iter__(self):
        return self

    def next(self):
        return (self.data.next(), self.labels.next())
        
            
def split_data(batch_iterator,
               splits = [0.8],
               rng_seed=888,
               in_memory=False,
               iterate=True,
               h5_path='/data/amazon/data.hd5',
               overwrite_previous=False):
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
            iterate --
                return data as an iterable? Only used if in_memory = True.
                Produces behavior akin to in_memory=False, where each slice 
                is returned as an iterator over (data, label) pairs.
            h5_path -- path to HDF5 file. Only used if in_memory is False
            overwrite_previous -- if h5_path is already a readable file,
                overwrite it?
            
        @Returns
            A list of iterables, where each iterable represents a 
            slice of the data and generates (data, label) pairs
            over individual records.
            If in_memory is False, this will be a list
            of H5Iterators, with each H5Iterator representing a
            slice of the data, yielding (data, labels).
            If in_memory is True and iterate is False, 
            this function will return  a list of 2-tuples of numpy arrays,
            where each 2-tuple is all of (data, labels) for each slice.
            
        
    '''

    # How many chunks to split into?
    nb_slices = len(splits) + 1
    np.random.seed(rng_seed)
    if in_memory:
        data_bins = None
        bin_sizes = [0]*nb_slices
        for data, labels in batch_iterator:
            bin_i = pick_splits(splits)
            if data_bins == None:
                data_bins = [ (np.ndarray(((0,) + data.shape[1:])), 
                              np.ndarray(((0,) + labels.shape[1:]))) 
                              for a in range(nb_slices) ]
            # store batch in the proper bin, creating numpy arrays
            # for data and labels if needed
            data_bins[bin_i] = (np.concatenate(
                                (data_bins[bin_i][0], data)),
                               np.concatenate(
                                (data_bins[bin_i][1], labels)))
            bin_sizes[bin_i] += data.shape[0]
        if iterate == True:
            for bin_i in range(nb_slices):
                bin = data_bins[bin_i]
                data_bins[bin_i] = iter(zip(iter(bin[0]), iter(bin[1])))
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
                first_bin_id = pick_splits(splits)
                logger.debug("Data shape: {}".format(first_data.shape))
                data_i = {}
                data_i[first_bin_id] = first_data.shape[0]
                h5_file["data_" + bin_names[first_bin_id]][0:data_i[first_bin_id]] = first_data
                h5_file["labels_" + bin_names[first_bin_id]][0:data_i[first_bin_id]] = first_labels
                # then do rest
                for new_data, new_labels in batch_iterator:
                    assert new_data.shape[0] == new_labels.shape[0]
                    # pick which bin to assign data to
                    bin_id = pick_splits(splits)
                    bin_name = bin_names[bin_id]
                    # get slice indexes
                    start_i = data_i.get(bin_id, 0)
                    end_i = start_i + new_data.shape[0]
                    # resize HDF5 datasets
                    h5_file["data_" + bin_name].resize(end_i, 0)
                    h5_file["labels_" + bin_name].resize(end_i, 0)
                    # write data
                    h5_file["data_" + bin_name][start_i:end_i, ...] = new_data
                    h5_file["labels_" + bin_name][start_i:end_i, ...] = new_labels
                    data_i[bin_id] = data_i.get(bin_id,0) + new_data.shape[0]
                    #logger.debug("Data shape: {}".format(new_data.shape))
                    #logger.debug("Output data shape: {}".format(h5_file["data_" + bin_name].shape))
                    logger.debug("Wrote from {} to {}.\nData bookmarks: {}".format(start_i, end_i, data_i))
        else:
            # fill in counts of each data slice
            data_i = {}
            with h5py.File(h5_path, "r") as f:
                for bin_i in range(nb_slices):
                    try:
                        data_i[bin_i] = f['data_' + str(bin_i)].shape[0]
                    except KeyError:
                        pass
                

        #now to return iterators over the HDF5 datasets for each slice
        # these can, in turn, be batched with batch_data (auughhh)
        data_iterators = []
        for bin_i in range(nb_slices):
            data_iterators.append((H5Iterator(h5_path, "data_" + str(bin_i), "labels_" + str(bin_i))))
        bin_sizes = [ data_i.get(i, 0) for i in range(nb_slices) ]
        return data_iterators, bin_sizes
        
               
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
