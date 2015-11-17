#!/usr/bin/env python
""" Contains some useful callbacks for
model serialization, logging, and 
evaluation of neon models"""

import os
import json
import time
import datetime
import logging
logger = logging.getLogger(__name__)
import numpy as np
import neon
import neon.callbacks
import neon.callbacks.callbacks
import neon.transforms.cost
#import matplotlib
#import sklearn.metrics

class NeonCallback(neon.callbacks.callbacks.Callback):
    """ This Callback object:
        -- saves performance metrics after every minibatch
            - cost
        -- after every epoch
            - train time
            - testing accuracy
            - testing confusion matrix

    """
    def __init__(self, model, train_data, test_data, save_path):
        self.conf_matrix_binary = ConfusionMatrixBinary()
        super(NeonCallback, self).__init__()
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.old_cost = self.be.zeros((1,1))
        self.save_path = save_path
        self.costs = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_confusions = []
        self.train_times = {}
        self.epoch_times = []
    
    @staticmethod
    def write_to_json(obj, path_base, path_decorator):
        """
        Writes (or overwrites) a JSON file located via a regular 
        combination of path base and path_decorator, dumping a JSON
        representation of obj to that file. Utility function for various 
        callback events.
        """
        # path decorator goes between filename and extension
        path_part, ext = os.path.splitext(path_base)
        full_path = "{}{}{}".format(path_part, path_decorator, ext)
        with open(full_path, "w") as f:
            json.dump(obj, f)

    def on_train_begin(self, epochs):
        # get start time for training
        self.train_times['start'] = time.time()

    def on_train_end(self):
        # write out start and end times for training
        self.train_times['end'] = time.time()
        self.write_to_json(self.train_times, self.save_path, "_traintimes")        
        
    def on_minibatch_end(self, epoch, minibatch):
        """Get costs per minibatch and
        write them to disk in JSON format every number
        of minibatches
        """
        # get cost
        new_cost = self.model.total_cost - self.old_cost
        # fiddle with neon's MOP abstractions to get the metric out
        cost_container = self.be.zeros((1,1))
        cost_container[:] = new_cost
        new_cost_scalar = float(cost_container.get()[0,0])
        #logger.debug(new_cost_scalar)
        # add to costs structure
        if epoch >= len(self.costs):
            self.costs.append([])
            logger.debug("Epoch {}, adding to costs list, now len {}".format(epoch,len(self.costs)))
        self.costs[epoch].append(new_cost_scalar)
        # save total cost to placeholder to compute difference in next batch
        self.old_cost[:] = self.model.total_cost
        # serialize every so often
        if len(self.costs[epoch]) % 1 == 0:
            self.write_to_json(self.costs, self.save_path, "_costs")

    def on_epoch_begin(self, epoch):
        # get start time for epoch
        self.epoch_times.append({})
        self.epoch_times[epoch]['start'] = time.time()

    def on_epoch_end(self, epoch):
        """Get train/test accuracy, produce
        epoch-wide charts of loss per minibatch"""
        # Get end time for training
        self.epoch_times[epoch]['end'] = time.time()
        # get accuracy scores
        logger.info("Computing confusions")
        test_confusion = self.conf_matrix_binary.get(self.model, self.test_data)
        # Training accuracy is really slow
        #logger.info("Computing training accuracy")
        #train_accuracy = self.model.eval(self.train_data, neon.transforms.Accuracy()).tolist()
        logger.info("Computing testing accuracy")
        #test_accuracy = self.model.eval(self.test_data, neon.transforms.Accuracy()).tolist()
        test_accuracy = float(test_confusion['tn'] + test_confusion['tp']) / float(sum(test_confusion.values()))
        # append and serialize to disk
        #self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        self.test_confusions.append(test_confusion)
        train_test_acc = { 'test' : self.test_accuracies }
        self.write_to_json(train_test_acc, self.save_path, "_accuracies")
        self.write_to_json(self.test_confusions, self.save_path, "_confusions")
        # finish writing costs to disk
        self.write_to_json(self.costs, self.save_path, "_costs")
        self.write_to_json(self.epoch_times, self.save_path, "_epochtimes")

        self.old_cost[:] = 0

class NeonCallbacks(neon.callbacks.callbacks.Callbacks):
    def add_neon_callback(self, metrics_path, **kwargs):
        self.add_callback(NeonCallback(self.model,
                                       self.train_set,
                                       self.valid_set,
                                       metrics_path), **kwargs)
    def __init__(self, model, train_set, output_file=None, valid_set=None,
                 valid_freq=None, progress_bar=True):
        super(NeonCallbacks, self).__init__(model, train_set, output_file,
            valid_set, valid_freq, progress_bar)
        self.valid_set = valid_set

class ConfusionMatrixBinary(neon.transforms.cost.Metric):

    """
    Compute the confusion matrix for a binary problem
    """

    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.matches = self.be.iobuf(1)
        self.outputs = self.be.zeros((2,2))
        self.metric_names = ['ConfusionMatrixBinary']
        logger.debug("Initting ConfusionMatrixBinary metric")

    def __call__(self, y, t):
        """
        Compute the confusion matrix metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            dict: Returns the metric
        """
        # convert back from onehot and compare
        #logger.debug("Fitting a ... {} shaped thing".format(y.shape))
        # if outputs are arrays
        if y.shape[0] > 1:
            self.preds[:] = self.be.argmax(y, axis=0)
            self.hyps[:] = self.be.argmax(t, axis=0)
        # if outputs are scalars
        else:
            # in case of single-neuron output (not onehot)
            self.preds[:] = y
            self.preds[:] = np.around(self.preds.get())
            self.hyps[:] = t

        self.matches[:] = self.be.equal(self.preds, self.hyps)

        conf_matrix = dict()
        predictions = self.preds.get().astype(bool)
        truth = self.hyps.get().astype(bool)
        matches = self.matches.get().astype(bool)

        #logger.debug("predictions: {}\ntruth: {}\nmatches: {}".format(predictions, truth, matches))
        
        # true positives
        conf_matrix['tp'] = np.sum(matches & truth)
        # true negatives
        conf_matrix['tn'] = np.sum(matches & np.logical_not(truth))
        # false positives
        conf_matrix['fp'] = np.sum(np.logical_not(matches) & np.logical_not(truth))
        # false negatives
        conf_matrix['fn'] = np.sum(np.logical_not(matches) & truth)

        self.outputs[:] = np.array([[conf_matrix['tp'], conf_matrix['fp']],[conf_matrix['fn'], conf_matrix['tn']]])

        return conf_matrix

    def get(self, model, data):
        """
        neon's ordinary metric interface is too convoluted for ordinary
        mortals, so this function takes a model and some data
        and calculates the accuracy on that data.
        """
        data.reset()
        conf_matrix = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
        running_sums = [0,0]
        for x,t in data:
            y = model.fprop(x, inference=True)
            #answers_sample = y.get()[:,:10].tolist()
            #truth_sample = t.get()[:,:10].tolist()
            #answers_sample = answers_sample + truth_sample
            #answers_sample = zip(*answers_sample)
            #logger.debug(answers_sample)
            new_conf_matrix = self(y, t)
            conf_matrix = { a: conf_matrix[a] + new_conf_matrix[a] for a in conf_matrix.keys() }
            running_sums[0] += np.sum([1 for a in np.nditer(t.get()) if a == 1.])
            running_sums[1] += np.sum([1 for a in np.nditer(t.get()) if a == 0.])
        return conf_matrix


class Accuracy(neon.transforms.cost.Metric):
    """
    Compute the accuracy error metric
    """
    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['Accuracy']

    def __call__(self, y, t):
        """
        Compute the accuracy error metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        if y.shape[0] == 1:
            # use labels
            self.preds[:] = np.around(y.get())
            self.hyps[:] = np.around(t.get())
        else:
            # convert back from onehot
            self.preds[:] = self.be.argmax(y, axis=0)
            self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.equal(self.preds, self.hyps)
        return self.outputs.get().mean()