#!/usr/bin/env python

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--start", default=25, type=int)
arg_parser.add_argument("cost_file", default="metrics_costs.json", nargs="?")
args = arg_parser.parse_args()

def plot_costs(json_path):
    with open(json_path) as f:
        json_obj = json.load(f)
        #df = np.array(json_obj)
        for idx, epoch in enumerate(json_obj):
            print idx, ":"
            costs_epoch = np.array(list(enumerate(epoch)))
            plt.figure()
            plt.plot(costs_epoch[args.start:,0], costs_epoch[args.start:,1])
            plt.savefig("costs_{}.png".format(idx))
            plt.close()

if __name__=="__main__":
    plot_costs(args.cost_file)

