#!/usr/bin/env python
""" Contains some useful callbacks for
model serialization, logging, and 
evaluation of neon models"""

import os
import json
import neon
#import matplotlib
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
        self.old_cost = self.be.zeros((1,1))
        self.save_path = save_path
        #self.minibatch_freq = minibatch_freq
        self.costs = []
        self.train_accuracies = []
        self.test_accuracies = []
    

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
        
    def on_minibatch_end(self, epoch, minibatch):
        """Check if it's time to do our metrics!"""
        # get cost
        new_cost = self.model.total_cost - self.old_cost
        # fiddle with neon's MOP abstractions to get the metric out
        cost_container = self.be.zeros((1,1))
        cost_container[:] = new_cost
        new_cost_scalar = float(cost_container.get()[0,0])
        #logger.debug(new_cost_scalar)
        # add to costs structure
        if epoch > len(self.costs):
            self.costs.append([])
            logger.debug("Epoch {}, adding to costs list, now len {}".format(epoch,len(self.costs)))
        self.costs[epoch].append(new_cost_scalar)
        # save total cost to placeholder to compute difference in next batch
        self.old_cost[:] = self.model.total_cost
        # serialize every so often
        if len(self.costs[epoch]) % 100 == 0:
            self.write_to_json(self.costs, self.save_path, "_costs")

    def on_epoch_end(self, epoch, epochs):
        """Get train/test accuracy, produce
        epoch-wide charts of loss per minibatch"""
        # get accuracy scores
        train_accuracy = self.model.eval(self.train_data, neon.transforms.Accuracy())
        test_accuracy = self.model.eval(self.test_data, neon.transforms.Accuracy())
        test_confusion = self.model.eval(self.test_data, ConfusionMatrixBinary())
        # append and serialize
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        train_test_acc = { 'train': self.train_accuracies,
                           'test' : self.test_accuracies }
        self.write_to_json(train_test_acc, self.save_path, "_accuracies")
        self.write_to_json(test_confusion, self.save_path, "_confusions")
        # finish writing costs to disk
        self.write_to_json(self.costs, self.save_path, "_costs")
        # TODO:  plot loss over the epoch
        

            
            

class NeonCallbacks(neon.callbacks.callbacks.Callbacks):
    def add_neon_callback(self, metrics_path):
        self.add_callback(NeonCallback(self.model,
                                       self.train_set,
                                       self.valid_set,
                                       metrics_path))
                                       

class ConfusionMatrixBinary(neon.transforms.cost.Metric):

    """
    Compute the confusion matrix for a binary problem
    """

    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['ConfusionMatrixBinary']

    def __call__(self, y, t):
        """
        Compute the accuracy metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy.array: Returns the metric
        """
        # convert back from onehot and compare
        self.preds[:] = self.be.argmax(y, axis=0)
        self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.equal(self.preds, self.hyps)

        conf_matrix = dict()
        predictions = self.preds.get()
        truth = self.hyps.get()
        matches = self.outputs.get()
        
        # true positives
        conf_matrix['tp'] = np.sum(matches & truth)
        # true negatives
        conf_matrix['tn'] = np.sum(matches & np.logical_not(truth))
        # false positives
        conf_matrix['fp'] = np.sum(np.logical_not(matches) & np.logical_not(truth))
        # false negatives
        conf_matrix['fn'] = np.sum(np.logical_not(matches) & truth)

        return conf_matrix

    def get(self, model, data):
        data.reset()
        conf_matrix = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
        running_sums = [0,0]
        for x,t in data:
            y = model.fprop(x, t)
            new_conf_matrix = self(y, t)
            conf_matrix = { a: conf_matrix[a] + new_conf_matrix[a] for a in conf_matrix.keys() }
            running_sums[0] += np.sum([1 for a in np.nditer(t.get()) if a == 1.])
            running_sums[1] += np.sum([1 for a in np.nditer(t.get()) if a == 0.])
        return conf_matrix

