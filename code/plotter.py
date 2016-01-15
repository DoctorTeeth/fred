#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cPickle
import logging

"""
The goal is to plot serialized runs after the fact.
What we want is to take as arguments a list of runFiles.
Where a runfile contains the xs (the number of steps)
and ys (the cost at that step) and the label for the graph.

So the one task we have is:
read this out and generate the plot.
"""
def generate_cost_graph(files, labels):

    logging.info("Graphing cost using files:")
    for f in files:
        logging.info("\t%s", f)

    for f, l in zip(files, labels):
        with open(f, 'rb') as pickle_file:
            read_vals = cPickle.load(pickle_file)
            results = read_vals['results']
            stamps = read_vals['time_stamps']

            y = np.array(results)
            x = np.array(stamps)

            plt.plot(x, y, label=l)

    plt.xlabel("Validation Runs")
    plt.ylabel("Validation Cost")
    plt.ylim(ymin=0, ymax=1)
    plt.legend(loc='upper right')
    plt.savefig('cost_graph')
    plt.close()

def generate_bandwidth_graph(files, labels, key, graph_name):

    logging.info("Graphing paramcopies using files:")
    for f in files:
        logging.info("\t%s", f)

    for f, l in zip(files, labels):
        with open(f, 'rb') as pickle_file:
            read_vals = cPickle.load(pickle_file)
            results = read_vals[key]
            stamps = range(len(results))

            y = np.cumsum(  np.array(results)  )
            x = np.array(stamps)

            plt.plot(x, y, label=l)

    plt.xlabel("Potential Copies")
    plt.ylabel("Actual Copies")
    plt.ylim(ymin=0, ymax=x[-1])
    plt.legend(loc='upper right')
    plt.savefig(graph_name)
    plt.close()

if __name__ == "__main__":
    fs = sys.argv[1:] # all args are filenames
    generate_cost_graph(fs, fs)
    generate_bandwidth_graph(fs, fs, 'parameter_copies', 'parameter_copies')
    generate_bandwidth_graph(fs, fs, 'gradient_copies', 'gradient_copies')
