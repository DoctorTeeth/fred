import theano
import theano.tensor as T
import numpy as np
from utils import param_copy

class HardSyncServer():
    """
    Completely synchronous gradient descent with N clients and batch size B.
    This should give exactly the same convergence as normal SGD with batch size B*N.
    """

    def __init__(self, model, clients, learning_rate=0.13):
        """
        Give server a copy of the model params.
        """
        self.learning_rate = learning_rate

        self.params = []
        for p in model.params:
            self.params.append(theano.shared(p.get_value()))

        self.timestamp = 0

        self.clients = clients
        self.pending_grads = {} # map client id to grad

    def apply_update(self, grads, timestamp, client):
        """
        Takes in a single gradient update and optionally applies it to the
        parameters.
        """

        unblock = False
        self.pending_grads[client] = grads

        if len(self.pending_grads) == self.clients:

            staleness = self.timestamp - timestamp

            # all grads is a list of lists of grads
            # each of which has the same length
            all_grads = self.pending_grads.values()

            # apply the param update
            for this_grad in all_grads:
                for g, p in zip(this_grad, self.params):
                    old_p = p.get_value()
                    p.set_value(old_p - self.learning_rate * (g / self.clients))

            # increment the timestamp, since weights have changed
            self.timestamp += 1
            unblock = True
            self.pending_grads = {}

        v = 1

        return self.params, self.timestamp, unblock, v

class SoftSyncServer():
    """
    Implements the n-softsync server described in that paper.
    """

    def __init__(self, model, clients, n, staleness_aware, learning_rate=0.13):
        """
        Give server a copy of the model params.
        n controls frequency of syncing
        n ranges from 1 to clients
        we wait for ~ (clients / n) updates before syncing
        """
        self.learning_rate = learning_rate
        self.staleness_aware = staleness_aware

        self.params = []
        for p in model.params:
            self.params.append(theano.shared(p.get_value()))

        self.timestamp = 0

        assert(n <= clients)
        self.c = int(clients / n)

        self.clients = clients
        self.pending_grads = []

    def apply_update(self, grads, timestamp, client):
        """
        Takes in a single gradient update and optionally applies it to the
        parameters.
        """

        unblock = False
        staleness = int(self.timestamp - timestamp)
        if not self.staleness_aware:
            staleness = 0

        self.pending_grads.append((staleness, grads))

        if len(self.pending_grads) == self.c:

            # apply the param update
            for this_stale, this_grad in self.pending_grads:
                modulation = ( 1.0 / (float(self.c) * float(this_stale + 1)))
                for g, p in zip(this_grad, self.params):
                    old_p = p.get_value()
                    p.set_value(old_p - self.learning_rate * g * modulation)

            # increment the timestamp, since weights have changed
            self.timestamp += 1
            unblock = True
            self.pending_grads = []

        v = 1
        
        return self.params, self.timestamp, unblock, v


class FASGDServer():
    """
    Implements the FASGD server.
    This is an improvement on the n-softsync server.
    """

    def __init__(self, model, clients, n, staleness_aware, learning_rate=0.13):
        """
        Give server a copy of the model params.
        n controls frequency of syncing
        n ranges from 1 to clients
        we wait for ~ (clients / n) updates before syncing
        """
        self.learning_rate = learning_rate
        self.staleness_aware = staleness_aware

        self.params = []
        self.magnitudes = []
        for p in model.params:
            self.params.append(theano.shared(p.get_value()))
            self.magnitudes.append(theano.shared(np.zeros_like(p.get_value())))

        self.timestamp = 0

        assert(n <= clients)
        self.c = int(clients / n)

        self.clients = clients
        self.pending_grads = []

    def update_estimates(self, grads):
        """
        update trailing estimates used for LR modulation.

        grads is just the list of grads for each of the parameters
        for 2 layer mlp, it would be W_1, W_2, b_1, b_2

        """

        for grad, mag in zip(grads, self.magnitudes):
            mag.set_value(mag.get_value() * 0.9 + np.abs(grad) * 0.1)


    def apply_update(self, grads, timestamp, client):
        """
        Takes in a single gradient update and optionally applies it to the
        parameters.

        Here we take a trailing estimate of the gradient
        magnitude (std deviation).
        and modulate by that as well

        we can choose to modulate per-parameter or not
        """

        unblock = False
        staleness = int(self.timestamp - timestamp)
        if not self.staleness_aware:
            staleness = 0

        self.update_estimates(grads)

        self.pending_grads.append((staleness, grads))

        epsilon = 0.001

        if len(self.pending_grads) == self.c:

            # apply the param update
            for this_stale, this_grad in self.pending_grads:
                modulation = ( 1.0 / (float(self.c) * float(this_stale + 1)))
                for g, p, m in zip(this_grad, self.params, self.magnitudes):
                    x = (1.0 / (m.get_value() + epsilon))
                    old_p = p.get_value()
                    p.set_value(old_p - self.learning_rate * g * modulation * x)

            # increment the timestamp, since weights have changed
            self.timestamp += 1
            unblock = True
            self.pending_grads = []

        # import pdb; pdb.set_trace()
        means = []
        sizes = []
        for mag in self.magnitudes:
            val = mag.get_value()
            means.append(np.mean(val))
            sizes.append(val.size)

        # TODO: any chance we should weight all tensors equally?
        v = 0
        t = np.sum(np.array(sizes))
        for m, s in zip(means, sizes):
            v = v + (float(s) / float(t)) * m

        return self.params, self.timestamp, unblock, v
