import numpy as np
import pickle
import os


def datagen_spiral(ns, nc):
    """
    The datagen_spiral function generates the spiral dataset.

    @type ns: int
    @param ns: the number of spirals
    @type nc: int
    @param nc: the number of classes
    @rtype: numpy.ndarray
    @return: the input data matrix X
    @rtype: numpy.ndarray
    @return:: the labels matrix Y
    """
    X = np.zeros((nc * ns, 2))
    Y = np.zeros((nc * ns))
    for c in range(nc):
        r = np.linspace(0, 1, ns) / 2
        t = np.linspace(c * 4, (c + 1) * 4, ns) + 0.15 * np.random.randn(1, ns)
        X[c * ns:(c + 1) * ns, 0] = r * np.sin(t)
        X[c * ns:(c + 1) * ns, 1] = r * np.cos(t)
        Y[c * ns:(c + 1) * ns] = c
    dataset = {'X': X, 'Y': Y}
    with open(os.path.join('data', 'spiral_dataset.pkl'), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return X, Y
