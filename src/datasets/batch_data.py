#!/usr/bin/env python

import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import numpy as np
import h5py

import data_utils


def batch_data(data_loader, batch_size=128, normalizer_fun=None, 
               transformer_fun=None, flatten=True):
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

    # set (near) identity functions for transformation functions when None
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
            #logger.debug("Normalization........")
            doc_text = normalizer_fun(doc_text)
            # transform document into a numpy array
            transformed_doc = transformer_fun(doc_text)
            docs.append(transformed_doc)
            labels.append(label)
        except data_utils.DataException as e:
            logger.debug("Type of input: {}".format(type(doc_text)))
            logger.info("{}: {}".format(type(e), e))

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


# Not fully tested -- still recommended to use
# batch_data
class BatchIterator:
    """Iterator class to wrap around batching functionality.
    Allows batched data to be iterated over multiple times.
    """
    def __init__(self, *args, **kwargs):
        """
        Args and kwargs should be arguments to batch_data.
        data_loader should be an iterator (i.e., able to be reused)
        """
        self.reset_args = args
        self.reset_kwargs = kwargs

    def __iter__(self):
        return batch_data(*self.reset_args, **self.reset_kwargs)

# not fully tested -- still recommended to use 
# xyz.load_data
class DataIterator:
    """Utility class to wrap a data loading generator function,
    providing a reusable data container if needed.
    """
    def __init__(self, load_fun, *args, **kwargs):
        """
        @Arguments:
            load_fun -- function object which returns
            a generator over tuples of individual records (data, label)

            args, kwargs are passed on to load_fun on every fresh iteration
        """
        self.load_fun = load_fun
        self.reset_args = args
        self.reset_kwargs = kwargs

    def __iter__(self):
        return self.load_fun(*self.reset_args, **self.reset_kwargs)


class H5Iterator:
    """Small utility class for iterating over an HDF5 file.
    Iterating over it yields tuples of (data, label) from datasets in the
    file with the names given in data_name and labels_name.
    By default, will randomly access records in any given iteration.
    """    
    def __init__(self, h5_path, data_name, labels_name, shuffle=True):
        """
        Arguments:
            h5_path -- path to HDF5 file to be accessed. This file should have the structure:
                / ...   [data_name]
                        [labels_name]
                        (...)
                where data_name and labels_name are datasets at the root level

            data_name -- name of features-bearing dataset

            labels_name -- name of dataset with labels. these may have a different type from 
                data_name, which is why they are stored separately

            shuffle -- should numpy randomly shuffle the indices each time an iterator is 
                made from this container?
        """
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
            # Singletons should be taken out of containers by default,
            # to match the behavior of the nnn.load_data class of functions
            # this presumes the data in the container is a string
            next_data = self.data[which_index]
            if next_data.shape == (1,):
                next_data = bytes(next_data[0])
                logger.debug("Going from singleton to string: '{}...'".format(next_data[::-1][:50]))
            else:
                logger.debug("H5 record shape: {}".format(next_data.shape))

            # take label out of container if singleton
            next_label = self.labels[which_index]
            if next_label.shape == (1,):
                next_label = next_label[0]
                logger.debug("Going from singleton to numeric type: {}".format(type(next_label)))
                #logger.debug("Shape: {}".format(next_label.shape))
            else:
                logger.debug("H5 record shape: {}".format(next_label.shape))
            yield (next_data, next_label)

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
               h5_path,
               splits = [0.8],
               rng_seed=None,
               overwrite_previous=False,
               shuffle=False):
    ''' Splits data into slices and returns a list of
        H5Iterators over each slice. Slice size is configurable.
        Probabilistic, so may not produce exactly the expected bin sizes, 
        especially for small data.
    
        @Arguments 
            batch_iterator --
                generator of tuples (data, label) where each of data, label
                is a numpy array with the first dimension representing batch size.
                This can be none if h5_path is valid and
                overwrite_previous=False (uses existing data, does not re-shuffle 
                or rearrange).
            h5_path -- path to HDF5 file
            splits --
                list of floats indicating how to split the data. The data will
                be split into len(splits) + 1 slices, with the final slice 
                having 1-sum(splits) of the data.
            rng_seed -- random number generator seed
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
        


def split_and_batch(data_loader, 
                    batch_size, 
                    doclength,
                    h5_path,
                    rng_seed=888,
                    normalizer_fun=lambda x: data_utils.normalize(x, max_length=doclength),
                    transformer_fun=lambda x: data_utils.to_one_hot(x)):
    """
    Convenience wrapper for most common splitting and batching
    workflow in neon. Splits data to an HDF5 path, if it does not already exist,
    and then returns functions for getting geerators over the datasets
    (gets around limitations of input to neon_utils.DiskDataIterator)
    """
    data_batches = batch_data(data_loader, batch_size,
        normalizer_fun=normalizer_fun,
        transformer_fun=None)
    (_, _), (train_size, test_size) = batch_data.split_data(imdb_batches, 
            h5_path, overwrite_previous=False, rng_seed=rng_seed)
    def train_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path=h5_path, overwrite_previous=False, shuffle=True)
        return batch_data(a,
            normalizer_fun=lambda x: x,
            transformer_fun=transformer_fun,
            flatten=True,
            batch_size=batch_size)
    def test_batcher():
        (a,b),(a_size,b_size)=split_data(None, h5_path, overwrite_previous=False,shuffle=False)
        return batch_data(b,
            normalizer_fun=lambda x: x,
            transformer_fun=transformer_fun,
            flatten=True,
            batch_size=batch_size)

    return (train_batcher, test_batcher), (train_size, test_size)               
                
if __name__=="__main__":
    # some demo code
    import imdb
    import amazon_reviews
    import data_utils
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_hdf5_demo', 
                        help="Demo basic batching. Requires a path to pre-split HDF5 file",
                        metavar="HDF5_PATH",
                        nargs='?')
    parser.add_argument('--iterator_demo',
                        metavar="AMAZON_JSON_GZ",
                        help='Demo iterator classes, requires a path to an Amazon json.gz file',
                        nargs='?')
    args = parser.parse_args()

    # Demo batching on pre-split data
    if args.batch_hdf5_demo:  
        print args.batch_hdf5_demo
        # get training and testing sets, and their sizes for amazon
        (amtr, amte), (amntr, amnte) = datasets, sizes = split_data(
            None, 
            h5_path=args.batch_hdf5_demo, 
            overwrite_previous=False,
            shuffle=True)
        import sys

        # get a record
        next_text, next_label = next(iter(amtr))

        try:
            print "Next record shape: {}".format(next_text.shape)
        except AttributeError as e:
            print "(No shape) Text: '{}'".format(next_text)


        # batch training, testing sets
        am_train_batch = batch_data(amtr,
            normalizer_fun=lambda x: data_utils.normalize(x, 
                max_length=300, 
                truncate_left=True),
            transformer_fun=None)
        am_test_batch = batch_data(amte,
            normalizer_fun=None,transformer_fun=None)

        # Spit out some sample data
        next_batch = am_train_batch.next()
        data, label = next_batch
        np.set_printoptions(threshold=np.nan)
        print "Batch properties:"
        print "Shape (data): {}".format(data.shape)
        print "Shape (label): {}".format(label.shape)
        print "Type: {}".format(type(data))
        print
        print "First record of first batch:"
        print "Type (1 level in): {}".format(type(data[0]))
        print "Type of record (2 levels in): {}".format(type(data[0,0]))
        print data[0,0]
        print "Sentiment label: {}".format(label[0,0])
        print "Data in numpy format:"
        oh = data_utils.to_one_hot(data[0,0])
        print np.array_str(np.argmax(oh,axis=0))
        print "Translated back into characters:\n"
        print ''.join(data_utils.from_one_hot(oh))


    # Demo iterator utility classes
    # iterate multiple times over same data 
    if args.iterator_demo:
        # Demo dataIterator class
        amz_iterator = DataIterator(amazon_reviews.load_data, args.iterator_demo)
        print "First run:"
        for i, (data, label) in enumerate(amz_iterator):
            print "{}: {}...".format(i, data[:50])
            if i >= 3: break
    
        print "Second run:"
        for i, (data, label) in enumerate(amz_iterator):
            print "{}: {}...".format(i, data[:50])
            if i >= 3: break
        # Demo batch iterator utility class
        amz_batches = BatchIterator(
            DataIterator(amazon_reviews.load_data, args.iterator_demo),
            batch_size=5, normalizer_fun=data_utils.normalize, 
            transformer_fun=lambda x: data_utils.to_one_hot(x), flatten=True)
        for i, (batch_data, batch_labell) in enumerate(amz_batches):
            print "Batch {}".format(i)
            print batch_data.shape
            print ''.join(data_utils.from_one_hot(batch_data[0].reshape(67, -1)))[::-1][:50], "..."
            if i >= 3: break

