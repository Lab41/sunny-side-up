#!/usr/bin/env python
""" Contains some useful callbacks for
model serialization, logging, and 
evaluation of neon models"""

import os
import json
import neon
import matplotlib
#import sklearn.metrics

class NeonCallback(neon.callbacks.callbacks.Callback):
    """ This Callback object:
        -- saves performance metrics after every N minibatches
            - cost
            - training accuracy
            - testing accuracy
            - precision/recall
            - confusion matrix
    """
    def __init__(self, model, train_data, test_data, save_path):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.old_cost = 0
        self.save_path = save_path
        #self.minibatch_freq = minibatch_freq
        self.costs = []
        self.train_accuracies = []
        self.test_accuracies = []
    

    def write_to_json(obj, path_base, path_decorator):
        path_part, ext = os.path.splitext(path_base)
        full_path = "{}{}{}".format(path_part, path_decorator, ext)
        with open(full_path, "w") as f:
            json.dump(obj, f)
        
    def on_minibatch_end(self, epoch, minibatch):
        """Check if it's time to do our metrics!"""
        # get cost
        new_cost = self.model.total_cost - self.old_cost
        # add to costs structure
        if epoch > len(self.costs):
            self.costs.append([])
        self.costs[epoch].append(new_cost)
        self.old_cost = new_cost
        # serialize every so often
        if len(self.costs[epoch]) % 1000 == 0:
            self.write_to_json(self.costs, self.save_path, "_costs")

    def on_epoch_end(self, epoch, epochs):
        """Get train/test accuracy, produce
        epoch-wide charts of loss per minibatch"""
        # get accuracy scores
        train_accuracy = self.model.eval(self.train_data, neon.transforms.Accuracy())
        test_accuracy = self.model.eval(self.test_data, neon.transforms.Accuracy())
        # append and serialize
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        accuracies_path_base, ext = 
        accuracies_path = "{}_accuracies{}".format(accuracies_path_base,ext)
        with open(accuracies_path, "w") as f:
            json.dump({'train': self.train_accuracies,
                       'test' : self.test_accuracies}, f)

        # finish writing costs to disk
        self.write_to_json(self.costs, self.save_path, "_costs")
        # TODO:  plot loss over the epoch
        

            
            

class NeonCallbacks(neon.callbacks.callbacks.Callbacks):
    def add_neon_callback(self, metrics_path):
        self.add_callback(NeonCallback(self.model,
                                       self.train_set,
                                       self.valid_set,
                                       metrics_path))
                                       

