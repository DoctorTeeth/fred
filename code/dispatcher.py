import numpy as np
from utils import param_copy, sample
import logistic_sgd as lg
import theano.tensor as T
import theano
import logging

def kPush(dispatcher, minibatch_index, client_id, j):
    """
    Generate the gradients no matter what.
    then decide whether we will send or coalesce
    then give back a timestamp signifying result of that decision
    """
    # TODO: try making this probabilistic

    potential_pushes = dispatcher.potential_pushes[client_id]

    # always make a new grad the first time a client is called
    fresh = dispatcher.parameter_stamps[client_id] == 0
    if potential_pushes % j == 0 or fresh:
        grads = dispatcher.get_grads(minibatch_index)
        parameter_timestamp = dispatcher.parameter_stamps[client_id]
        dispatcher.gradient_copies.append(1)

        # TODO: we can't just pass grads like that, can we?
        dispatcher.cached_grads[client_id] = (grads, parameter_timestamp)
    else:
        # TODO: might have to assign into grads in funky way
        grads, parameter_timestamp = dispatcher.cached_grads[client_id]
        dispatcher.gradient_copies.append(0)

    dispatcher.potential_pushes[client_id] = potential_pushes + 1

    return grads, parameter_timestamp

def kFuzzyPush(dispatcher, minibatch_index, client_id, j):
    """
    Generate the gradients no matter what.
    then decide whether we will send or coalesce
    then give back a timestamp signifying result of that decision
    """
    # TODO: try making this probabilistic
    r = np.random.uniform()
    epsilon = 0.0001
    c = dispatcher.cpush
    v = dispatcher.v
    rval = (1 / (1 + (c / (v + epsilon) ) ) )

    # import pdb; pdb.set_trace()
    potential_pushes = dispatcher.potential_pushes[client_id]

    # always make a new grad the first time a client is called
    fresh = dispatcher.parameter_stamps[client_id] == 0

    if r < rval or fresh:
        grads = dispatcher.get_grads(minibatch_index)
        parameter_timestamp = dispatcher.parameter_stamps[client_id]
        dispatcher.gradient_copies.append(1)

        # TODO: we can't just pass grads like that, can we?
        dispatcher.cached_grads[client_id] = (grads, parameter_timestamp)
    else:
        # TODO: might have to assign into grads in funky way
        grads, parameter_timestamp = dispatcher.cached_grads[client_id]
        dispatcher.gradient_copies.append(0)

    dispatcher.potential_pushes[client_id] = potential_pushes + 1

    return grads, parameter_timestamp


# TODO: will have to pass different versions of this
def kFetch(dispatcher, k, client_id):
    """
    Decide whether to grab params from the server

    get the count for number of potential updates for this
    client
    if number of potential updates mod k is 0, update,
    else keep params the same
    """
    potential_updates = dispatcher.potential_updates[client_id]
    if potential_updates % k == 0:
        dispatcher.parameter_copies.append(1)
        return_value = True
    else:
        dispatcher.parameter_copies.append(0)
        return_value = False

    dispatcher.potential_updates[client_id] = potential_updates + 1

    return return_value

def kFuzzyFetch(dispatcher, k, client_id):
    """
    decide whether to grab params fuzzily
    c is the tuning param
    """
    potential_updates = dispatcher.potential_updates[client_id]
    r = np.random.uniform()
    epsilon = 0.0001
    c = dispatcher.cfetch # a parameter
    v = dispatcher.v # not a parameter, a statistic
    rval = (1 / (1 + (c / (v + epsilon) ) ) )

    # import pdb; pdb.set_trace()
    if r < rval:
        dispatcher.parameter_copies.append(1)
        return_value = True
    else:
        dispatcher.parameter_copies.append(0)
        return_value = False

    dispatcher.potential_updates[client_id] = potential_updates + 1

    return return_value



# TODO: make the type of param_fetcher a parameter to __init__
class Dispatcher():
    """
    Manages overall simulation.
    """

    def __init__(self, clients, batch_size, validate_frequency=100, k=1,
                 fetcher=kFetch, cfetch=0.01, pusher=kPush, j=1, cpush=0.01):

        self.v = 1
        self.k = k
        self.j = j
        self.cfetch = cfetch
        self.cpush  = cpush
        self.fetcher = fetcher
        self.pusher = pusher

        self.cached_grads = {}

        self.client_priorities = np.ones((clients, 1))
        self.parameter_stamps = np.zeros((clients, 1))
        self.potential_updates = np.zeros((clients, 1))
        self.potential_pushes = np.zeros((clients, 1), dtype=np.int)

        # tracks last grad stamp sent by this client to server
        self.sent_grad_stamps = np.zeros((clients, 1))

        self.validation_timestamps = []
        self.validation_results = []

        # stats about parameter and grad copies
        # TODO: when we go to per-tensor, this gets more complicated
        self.parameter_copies = []
        self.gradient_copies = []


        valid_size = 600

        # represents set of clients currently blocking
        self.blocking = []

        # TODO: move a lot of this stuff out of dispatcher
        # it should live in a separate utility function
        dataset = 'mnist.pkl.gz'

        datasets = lg.load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]


        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / valid_size

        self.validate_frequency = validate_frequency

        # generate symbolic variables for input (x and y represent a minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        # construct the logistic regression class, MNIST imgs are 28*28
        self.classifier = lg.LogisticRegression(input=x, n_in=28 * 28, n_out=10)

        # initialize all client params to copies of the original params
        self.client_params = []
        for idx in range(clients):
            this_client_params = []
            for p in self.classifier.params:
                this_client_params.append(theano.shared(p.get_value()))
            self.client_params.append(this_client_params)

        # the cost the negative log likelihood of the model in symbolic format
        cost = self.classifier.negative_log_likelihood(y)

        index = T.lscalar()  # index to a [mini]batch
        grads = T.grad(cost, self.classifier.params)

        # function to do validation
        self.validate_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(y),
            givens={
                x: valid_set_x[index * valid_size: (index + 1) * valid_size],
                y: valid_set_y[index * valid_size: (index + 1) * valid_size]
            }
        )

        self.get_grads = theano.function(
            inputs=[index],
            outputs=grads,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def update_priorities(self):
        """
        Update priorities governing which client will be picked next.
        """
        # TODO: actually write this code
        pass


    def fetch_update(self, iteration):
        """
        Decide where the next grad update will come from.

        Also decides whether to drop the grad
        """

        client_id = sample(self.client_priorities, self.blocking)
        self.update_priorities()

        # overwrite the models params with client params
        param_copy(self.client_params[client_id], self.classifier.params)

        minibatch_index = iteration % self.n_train_batches
        grads, parameter_timestamp = self.pusher(self,
                                                            minibatch_index,
                                                            client_id,
                                                            j=self.j)

        return (grads, parameter_timestamp, client_id)

    def update_parameters(self, params, client, server_stamp, unblock, v):
        """
        Give new parameters to a client.
        """
        # update grad magnitude mean
        self.v = v

        # add client to blocking set no matter what
        if client not in self.blocking:
            self.blocking.append(client)

        if unblock: # update params of all blocking clients
            for client_id in self.blocking:

                if self.fetcher(self, self.k, client):
                    # we've decided to pull updates
                    param_copy(params, self.client_params[client_id])
                else:
                    # we're going to ignore that (we could have not copied it)
                    pass

                self.parameter_stamps[client_id] = server_stamp

            self.blocking = [] # nothing is blocking now

    def validate(self, params, server_stamp, unblock):
        """
        Optionally run validation policy.
        This may include all sorts of logging etc.
        """
        if unblock and (server_stamp) % self.validate_frequency == 0:
            # Make sure that we run validation with the server parameters
            param_copy(params, self.classifier.params)

            validation_losses = [self.validate_model(i)
                                    for i in xrange(self.n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            logging.info('server_stamp %i, validation error %f %%' %
                        ( server_stamp, this_validation_loss * 100.)
            )

            self.validation_timestamps.append(server_stamp)
            self.validation_results.append(this_validation_loss)
