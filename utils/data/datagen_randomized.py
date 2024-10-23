import numpy as np
import pickle


def datagen_randomized(m_gen, d, c, n, save=True):
    """
    The datagen_randomized function generates the randomized dataset of desired size using a 2-layer fully-connected
    ReLU generative neural network. The input data matrix is n x d and the labels matrix is n x c.

    @type m_gen: int
    @param m_gen: the number of neurons in the neural network
    @type d: int
    @param d: the number of input features
    @type c: int
    @param c: the number of output features
    @type n: int
    @param n: the number of data instances
    @type save: bool
    @param save: whether or not to save the randomized dataset to a pickle file
    @rtype: numpy.ndarray
    @return: the input data matrix X
    @rtype: numpy.ndarray
    @return:: the labels matrix Y
    """

    # Generate the randomized dataset
    U = np.random.randn(d, m_gen)
    V = np.random.randn(c, m_gen)
    X = np.random.randn(n, d)
    relu = lambda x: np.multiply(x > 0, x)
    Y = relu(X @ U) @ V.T
    return X, Y
