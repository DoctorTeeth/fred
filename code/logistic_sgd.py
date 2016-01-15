import cPickle
import gzip
import os
import sys
import numpy
import theano
import theano.tensor as T
import logging

# logging.basicConfig(level=logging.INFO)

class LogisticRegression(object):
    """Multi-class Logistic Regression Class, weight matrix W, bias b
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        input is a minibatch input
        :param n_in: number of input units
        :param n_out: number of output units

        """
        n_hidden = 100

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W_1 = theano.shared(
            value=numpy.random.randn(
                n_in, n_hidden
            ),
            name='W_1',
            borrow=True
        )

        self.W_2 = theano.shared(
            value=numpy.random.randn(
                n_hidden, n_out
            ),
            name='W_2',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b_1 = theano.shared(
            value=numpy.zeros(
                (n_hidden,),
                dtype=theano.config.floatX
            ),
            name='b_1',
            borrow=True
        )

        self.b_2 = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b_2',
            borrow=True
        )

        self.hidden = T.nnet.relu(T.dot(input, self.W_1) + self.b_1)

        # expression for computing the matrix of class-membership probs
        self.p_y_given_x = T.nnet.softmax(T.dot(self.hidden, self.W_2) + self.b_2)

        # compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W_1, self.W_2, self.b_1, self.b_2]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        y is a vector of labels
        """
        # the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch

        y is the target vector
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset from path pointed to by 'dataset' '''
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        urllib.urlretrieve(origin, dataset)


    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        We need this so we don't do too much HtoD when running on GPU
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # necessary for GPU interop
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

