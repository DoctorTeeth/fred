#!/usr/bin/env python

from run import run
import numpy as np
import argparse
from dispatcher import Dispatcher
from servers import HardSyncServer, SoftSyncServer
from utils import save_state
import cPickle
import sys
import logging


def assert_same(names):
    """
    return true if all files in names have same results.
    """
    assert(len(names) > 1)

    all_results = []
    for name in names:
        with open(name, 'rb') as pickle_file:
            read_vals = cPickle.load(pickle_file)
            results = read_vals['results']
            all_results.append(results)

    for result in all_results[1:]:
        if result != all_results[0]:
            return False

    return True


def test_1():

    """
    hard sync should be the same as synchronous, as long as the product of
    clients and batch size is constant
    """
    vf = 1

    # TODO: need to pass RNG to model?
    np.random.seed(0)
    clients1 = 1
    batch1 = 100
    d1 = Dispatcher(clients1, batch1, validate_frequency=vf)
    hard1 = HardSyncServer(d1.classifier, clients1)
    run(d1, hard1, 250, "/tmp/first")

    np.random.seed(0)
    clients2 = 4
    batch2 = 25
    d2 = Dispatcher(clients2, batch2, validate_frequency=vf)
    hard2 = HardSyncServer(d2.classifier, clients2)
    run(d2, hard2, 1000, "/tmp/second")

    return assert_same(["/tmp/first", "/tmp/second"])

def test_2():
    """
    soft sync with n = 1 is the same as hard sync
    given that we don't let a client keep going sans update
    """
    clients = 30
    bsz = 400
    vf = 1

    np.random.seed(0)
    d1 = Dispatcher(clients, bsz, validate_frequency=vf)
    soft = SoftSyncServer(d1.classifier, clients, 1, False)
    run(d1, soft, 250, "/tmp/first")

    np.random.seed(0)
    d2 = Dispatcher(clients, bsz, validate_frequency=vf)
    hard = HardSyncServer(d2.classifier, clients)
    run(d2, hard, 250, "/tmp/second")

    return assert_same(["/tmp/first", "/tmp/second"])


all_tests = [(test_1, "test_1"),
             (test_2, "test_2")]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, filename='test.log', filemode='w')
    logging.info("RUNNING TESTS")

    for test, name in all_tests:
        logging.info("TEST: %s", name)
        if not test():
            print "FAIL: ", name
            sys.exit(1)

    print "PASS"
    logging.info("TESTS FINISHED")
