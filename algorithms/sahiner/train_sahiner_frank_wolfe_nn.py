import time
import cvxpy as cp
import numpy as np
from algorithms.generic.nonneg_pca import nonneg_pca


def train_sahiner_frank_wolfe_nn(X, y, sign_patterns, beta, t, epochs=1, lr=0.5,
                                 use_cvxpy=False, print_freq=100, return_times=False):
    """
    The train_sahiner_frank_wolfe_nn function trains a two-layer fully-connected ReLU neural network (one hidden layer)
    using Sahiner's convex semi-infinite dual formulation of the NN training problem. Please see the following
    publication for more details.

    Sahiner, Arda, et al. "Vector-output relu neural network problems are copositive programs: Convex analysis of two
    layer networks and polynomial-time algorithms." arXiv preprint arXiv:2012.13329 (2020).

    Author(s): Arda Sahiner et al.

    @param A: the data matrix
    @param y_hot: the one-hot encoded labels
    @type: list
    @param sign_patterns: a list of sign patterns
    @param beta: float
    @param beta: the regularization parameter
    @type t: float
    @param t: the value of t (see Sahiner's paper for details)
    @type epochs: int
    @param epochs: the number of epochs to run Sahiner's FW algorithm
    @type lr: float
    @param lr: the parameter that specifies the weight for the convex FW update
    @type use_cvxpy: bool
    @param use_cvxpy: the flag that indicates whether to solve subproblem using CVXPY
    @type print_freq: int
    @param print_freq: how many iterations to wait till the loss is printed
    @param return_times: the flag that indicates whether to return timing information or the V matrix (solution)
    @rtype: numpy.ndarray
    @return: the objective value for the copositive relaxation
    @rtype: numpy.ndarray
    @return: the solution or the recorded times for each FW iteration (need to difference to get iteration time)
    """
    n = X.shape[0]
    d = X.shape[1]
    c = y.shape[1]
    P = len(sign_patterns)
    V = [np.zeros((d, c)) for i in range(2 * P)]
    const = 2
    times = np.zeros(epochs)
    losses = np.zeros(epochs)

    for ep in range(epochs):
        lr = const / (const + ep) ** (1)

        R_k = y - sum([np.multiply(sign_patterns[i], X) @ V[i] for i in range(P)])
        loss = 1 / 2 * np.linalg.norm(R_k) ** 2 + beta * t

        losses[ep] = loss
        times[ep] = time.time()

        if ep % print_freq == 0:
            print(ep, loss)

        max_nonneg_pca = np.zeros(P)
        max_v_pca = []

        for i in range(P):
            mask = np.multiply(sign_patterns[i], X)
            R = 2 * mask - X
            M = mask.T @ R_k @ R_k.T @ mask

            u, value = nonneg_pca(M, R)
            u = np.real(u)
            value = np.real(value)

            g = R_k.T @ mask @ u
            if np.linalg.norm(g) != 0:
                g = g / np.linalg.norm(g)

            max_nonneg_pca[i] = value
            max_v_pca.append(np.outer(u, g))

        best = np.argmax(max_nonneg_pca)
        v_chosen = max_v_pca[best]
        other_indices = np.delete(np.arange(P), best)

        if use_cvxpy:
            step_size = cp.Variable(1)
            V_next = [(1 - step_size) * V[i] + step_size * t * v_chosen if i == best else V[i] * (1 - step_size) for i
                      in range(P)]
            res_next = y - sum([sign_patterns[i] * X @ V_next[i] for i in range(P)])
            obj = cp.Minimize(cp.norm(res_next) ** 2)
            constr = [step_size >= 0, step_size <= 1]
            prob = cp.Problem(obj, constr)
            prob.solve()

            V = [V_next[i].value for i in range(P)]
            if ep % print_freq == 0:
                print(step_size.value)
        else:
            V = [(1 - lr) * V[i] + lr * t * v_chosen if i == best else V[i] * (1 - lr) for i in range(P)]

    R_k = y - sum([np.multiply(sign_patterns[i], X) @ V[i] for i in range(P)])
    loss = 1 / 2 * np.linalg.norm(R_k) ** 2 + beta * np.sum([np.linalg.norm(V[i], 'nuc') for i in range(P)])

    if return_times:
        return losses, times
    else:
        return loss, V