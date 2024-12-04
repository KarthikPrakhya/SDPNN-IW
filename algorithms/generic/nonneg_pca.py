import numpy as np


def nonneg_pca(M, X):
    """
    Solves the non-negative PCA problem. This is used as a helper function for train_sahiner_frank_wolfe_nn function to
    compute the Linear Minimization Oracle (LMO) for Sahiner's FW algorithm.

    Author(s): Arda Sahiner et al.

    Please see the following publications for more details.

    Asteris, Megasthenis, Dimitris Papailiopoulos, and Alexandros Dimakis. "Nonnegative sparse PCA with provable
    guarantees." International Conference on Machine Learning. PMLR, 2014.

    Sahiner, Arda, et al. "Vector-output relu neural network problems are copositive programs: Convex analysis of two
    layer networks and polynomial-time algorithms." arXiv preprint arXiv:2012.13329 (2020).

    This function solves the following problem:

    min_u u^T M u

    s.t. Xu \geq 0
    norm_2(u) \leq 1

    @type M: numpy.ndarray
    @param M: the M matrix for non-negative PCA problem
    @type X: numpy.ndarray
    @param X: the X matrix for non-negative PCA problem
    @rtype: numpy.ndarray
    @return: the solution to non-negative PCA problem
    """
    n = X.shape[0]
    d = X.shape[1]

    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals)

    u1 = vecs[:, idx]

    if np.all(X @ u1 >= 0):
        return u1, vals[idx]
    elif np.all(X @ u1 <= 0):
        return -u1, vals[idx]

    elif d == 2:
        C = []
        for i in range(n):
            ci = np.array([-X[i, 1], X[i, 0]]) / (np.linalg.norm(X[i, :]) + 1e-12)
            if np.all(X @ ci >= 0):
                C.append(ci)
            elif np.all(X @ ci <= 0):
                C.append(-ci)

        if len(C) == 0:
            return np.zeros(d), 0.0

        quads = np.array([C[i].T @ M @ C[i] for i in range(len(C))])
        idx = np.argmax(quads)
        return C[idx], quads[idx]

    C = []
    for i in range(n):
        j = min([l for l in range(d) if X[i, l] != 0])

        top = np.hstack((np.eye(j), np.zeros((j, d - j - 1))))
        mid = -1 / (X[i, j]) * np.delete(X[i], j)
        bottom = np.hstack((np.zeros((d - j - 1, j)), np.eye(d - j - 1)))

        H = np.vstack((top, mid, bottom))

        U_h, S_h, V_h = np.linalg.svd(H, full_matrices=False)
        M_curr = U_h.T @ M @ U_h
        X_curr = np.delete(X, i, 0) @ U_h

        c_i_hat, _ = nonneg_pca(M_curr, X_curr)

        c_i = U_h @ c_i_hat
        C.append(c_i)

    quads = np.array([C[i].T @ M @ C[i] for i in range(len(C))])
    idx = np.argmax(quads)

    return C[idx], quads[idx]