#!/usr/bin/env python
import argparse
import json
import numpy as np

def summary_metrics(json_path):
    conf_matrices = json.load(open(json_path))
    epoch_metrics = []
    for epoch, conf_matrix in enumerate(conf_matrices):
        metrics = {}
        metrics['FN'] = float(conf_matrix['fn'])
        metrics['FP'] = float(conf_matrix['fp'])
        metrics['TN'] = float(conf_matrix['tn'])
        metrics['TP'] = float(conf_matrix['tp'])
        metrics['accuracy'] = (metrics['TP'] + metrics['TN']) / \
            (np.sum([val for val in metrics.itervalues()]))
        metrics['precision'] = metrics['TP'] / (metrics['TP'] +
                                metrics['FP'])
        metrics['recall'] = metrics['TP'] / (metrics['TP'] + 
                                metrics['FN'])
        metrics['f1'] = 2 * ((metrics['precision'] * metrics['recall'])/
                             (metrics['precision'] + metrics['recall']))
        epoch_metrics.append(metrics)
    # take the epoch with the best F1 as best
    best_epoch = np.argmax([e['f1'] for e in epoch_metrics])
    best_metrics = epoch_metrics[best_epoch]
    best_metrics['epoch'] = best_epoch
    return best_metrics

def summary_times(epochtimes_json_path, traintimes_json_path):
    train_times = json.load(open(traintimes_json_path))
    epoch_times = json.load(open(epochtimes_json_path))
    data_loading_time = epoch_times[0]['start'] - train_times['start']
    epoch_training_times = [ e['end'] - e['start'] for e in epoch_times ]
    epoch_testing_times = [ epoch_times[i]['start'] - epoch_times[i-1]['end'] for i in range(1, len(epoch_times)) ]
    return { 
            'training_per_epoch_secs'       : epoch_training_times,
            'testing_per_epoch_secs'        : epoch_testing_times,
            'data_loading_secs'             : data_loading_time,
            'total_time'                    : train_times['end'] - train_times['start']
           }

def main(confusions, epoch_times, train_times):
    results_hash = { 'metrics' : summary_metrics(confusions),
                     'times'   : summary_times(epoch_times, train_times) }
    results_json = json.dumps(results_hash, indent=2)
    return results_json

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--confusions", nargs=1, default="metrics_confusions.json")
    arg_parser.add_argument("--times", nargs=2, 
        metavar=("EPOCHTIMES","TRAINTIMES"), 
        default=["metrics_epochtimes.json", "metrics_traintimes.json"])

    args = arg_parser.parse_args()
    main(args.confusions, args.times[0], args.times[1])
    results_hash = { 'metrics' : summary_metrics(args.confusions),
                     'times'   : summary_times(*args.times) }
    results_json = json.dumps(results_hash, indent=2)
    print results_json
