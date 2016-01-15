#!/usr/bin/env python

import numpy as np
import argparse
import logging
from dispatcher import Dispatcher, kFuzzyFetch, kFetch, kPush, kFuzzyPush
from servers import HardSyncServer, SoftSyncServer, FASGDServer
from utils import save_state

"""
Harness for simulating distributed training in a deterministic, single-node context.
"""

def run(dispatcher, server, iterations, filename):
    """
    Run the experiment using provided dispatcher and server
    """

    for iteration in range(iterations):

        # fetch the results of a single client run
        (grads, client_stamp, client) = dispatcher.fetch_update(iteration)

        # give those results to the parameter server
        params, server_stamp, unblock, v = server.apply_update(grads,
                                                               client_stamp,
                                                               client)

        # give the (potentially) new parameters back to that client
        # all other clients are by definition still working
        # TODO: should client wait for potential param update?
        dispatcher.update_parameters(params, client, server_stamp, unblock, v)

        # optionally run validation and report stats
        dispatcher.validate(params, server_stamp, unblock)

    save_state(dispatcher, filename)

if __name__ == "__main__":

    # configure the logger
    # logging.basicConfig(filename='fasgd.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO, filename='fasgd.log', filemode='w')

    logging.info("CONFIGURE TRAINING OPTIONS")

    np.random.seed(0)
    rng = np.random.RandomState(0)

    parser = argparse.ArgumentParser(description="FASGD")
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=830)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--filename", default="out.pkl")
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    # TODO: get rid of args that should be in configs instead
    # TODO: clean this up, it's unnecessarily complicated.
    clients = 10000
    bsz = 128
    d1 = Dispatcher(clients, bsz)
    d2 = Dispatcher(clients, bsz)
    # d3 = Dispatcher(clients, bsz)
    # d4 = Dispatcher(clients, bsz)
    # d5 = Dispatcher(clients, bsz)
    # d6 = Dispatcher(clients, bsz)
    # d7 = Dispatcher(clients, bsz)
    # d8 = Dispatcher(clients, bsz)

    lr = 0.005
    fasgd_1 = FASGDServer(d1.classifier, clients, clients, True, learning_rate=0.005)
    fasgd_2 = SoftSyncServer(d2.classifier, clients, clients, True, learning_rate=0.04)
    # fasgd_3 = FASGDServer(d3.classifier, clients, clients, True, learning_rate=0.02)
    # fasgd_4 = FASGDServer(d4.classifier, clients, clients, True, learning_rate=0.04)
    # fasgd_5 = FASGDServer(d5.classifier, clients, clients, True, learning_rate=0.08)
    # fasgd_6 = FASGDServer(d6.classifier, clients, clients, True, learning_rate=0.16)
    # fasgd_7 = FASGDServer(d7.classifier, clients, clients, True, learning_rate=0.32)
    # fasgd_8 = FASGDServer(d8.classifier, clients, clients, True, learning_rate=0.64)

    server_configs = [(fasgd_1, "fasgd_0.005_10000_128", d1),
                      (fasgd_2, "staleness_aware_0.04_10000_128", d2)]
                      # (fasgd_3, "fasgd2_0.02", d3),
                      # (fasgd_4, "fasgd2_0.04", d4),
                      # (fasgd_5, "fasgd2_0.08", d5),
                      # (fasgd_6, "fasgd2_0.16", d6),
                      # (fasgd_7, "fasgd2_0.32", d7),
                      # (fasgd_8, "fasgd2_0.64", d8)]

    output_names = []
    labels = []
    for config, label, d in server_configs:
        name = "outputs/" + label + "_" + args.filename
        output_names.append(name)
        labels.append(label)
        run(d, config, args.iterations, name)

    import plotter
    plotter.generate_cost_graph(output_names, labels)
    plotter.generate_bandwidth_graph(output_names, labels,
                                     'parameter_copies', 'parameter_copies')
    plotter.generate_bandwidth_graph(output_names, labels,
                                     'gradient_copies', 'gradient_copies')
