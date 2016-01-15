import numpy as np
import cPickle

"""
Miscellaneous utilities used for simulations.
"""

def param_copy(src, dst):
    for src_param, dst_param in zip(src, dst):
        dst_param.set_value(src_param.get_value())

def sample(vec, blocking):
    """
    vec is just any vector of value that we'll softmax sample from
    """
    idx = -1
    while idx < 0 or idx in blocking:

        softmax = np.exp(vec) / np.sum(np.exp(vec))
        r = np.random.uniform(0,1)
        sums = np.cumsum(softmax)
        w = np.where(r < sums)
        if w == []:
            idx = 0
        else:
            idx = w[0][0]

    return idx

def save_state(dispatcher, filename):

    save_vals = {'parameters': [p.get_value() for p in dispatcher.classifier.params],
                 'time_stamps': dispatcher.validation_timestamps,
                 'parameter_copies': dispatcher.parameter_copies,
                 'gradient_copies': dispatcher.gradient_copies,
                 'results': dispatcher.validation_results}

    cPickle.dump(save_vals,
                    file(filename, 'wb'),
                    cPickle.HIGHEST_PROTOCOL)
