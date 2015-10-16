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
            docs_np = np.array(docs)
            if flatten==True:
                # transform to form (batch_size, w*h); flattening doc
                docs_np = docs_np.reshape(batch_size,-1)
            # labels come out in a separate (batch_size, 1) np array
            labels_np = np.array(labels).reshape(batch_size, -1)
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
    # TODO: implement resetting
    def __init__(self, h5_path, data_name, labels_name):
        self.h5file = h5py.File(h5_path, "r")
        self.data = iter(self.h5file[data_name])
        self.labels = iter(self.h5file[labels_name])
        
    def __del__(self):
        self.h5file.close()
        
    def next(self):
        return (self.data.next(), self.labels.next())
        
            
def split_data(batch_iterator,
               splits = [0.2],
               in_memory=False,
               h5_path='/data/amazon/data.hd5',
               overwrite_previous=False):
    ''' Splits data into slices and returns a tuple of
        iterators over each slice. Slice size is configurable.
        Probabilistic, so may not produce exactly the expected bin sizes, 
        especially for small data.
    
        @Arguments 
            batch_iterator --
        
            splits --
            
            in_memory --
            
            amazon_url --
            
            h5dir --
            
            overwrite_previous --
            
        @Returns
        
    '''

    # How many chunks to split into?
    nb_slices = len(splits) + 1
    
    if in_memory:
        data_bins = { str(i): {} for i in range(nb_slices) }
        for data, labels in batch_iterator:
            bin_i = str(pick_splits(splits))
            # store batch in the proper bin, creating numpy arrays
            # for data and labels if needed
            data_bins[bin_i]["data"] = np.concatenate(
                (data_bins[bin_i].get("data", 
                                      np.ndarray(((0,) + data.shape[1:]))), 
                data))
            data_bins[bin_i]["labels"] = np.concatenate(
                (data_bins[bin_i].get("labels", 
                                      np.ndarray(((0,) + labels.shape[1:]))), 
                labels))
        return data_bins
                    
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
                    logger.debug("Data shape: {}".format(new_data.shape))
                    logger.debug("Output data shape: {}".format(h5_file["data_" + bin_name].shape))
                    logger.debug("Wrote from {} to {}.\nData bookmarks: {}".format(start_i, end_i, data_i))
                    
        # now to return iterators over the HDF5 datasets for each slice
        # these can, in turn, be batched with batch_data (auughhh)
        data_iterators = []
        for bin_i in range(nb_slices):
            data_iterators.append((H5Iterator(h5_path, "data_" + str(bin_i), "labels_" + str(bin_i))))
        return data_iterators
        
               
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
