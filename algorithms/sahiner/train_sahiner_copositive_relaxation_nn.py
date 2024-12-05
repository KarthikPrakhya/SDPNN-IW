import torch
import cvxpy as cp


def train_sahiner_copositive_relaxation_nn(A, y_hot, sign_patterns, beta, eps=1e-4):
    """
    The train_sahiner_copositive_relaxation_nn function trains a two-layer fully-connected ReLU neural network
    (one hidden layer) using Sahiner's copositive relaxation of the NN training problem. Please see the following
    publication for more details.

    Sahiner, Arda, et al. "Vector-output relu neural network problems are copositive programs: Convex analysis of two
    layer networks and polynomial-time algorithms." arXiv preprint arXiv:2012.13329 (2020).

    Author(s): Arda Sahiner et al.

    @param A: the data matrix
    @param y_hot: the one-hot encoded labels
    @type: list
    @param sign_patterns: a list of sign patterns
    @param beta: the regularization parameter
    @type eps: float
    @param eps: the eps argument for CVXPY
    @rtype: float
    @return: the objective value for the copositive relaxation
    @rtype: float
    @return: the solve time for the copositive relaxation
    """
    n = len(A)
    P = len(sign_patterns)
    d = A.shape[1]
    # y_hot = y_hot.data.numpy()
    U_list = [cp.Variable((d, d), PSD=True) for i in range(P)]
    masked_A = [sign_patterns[i] * A for i in range(P)]
    preds = sum([masked_A[i] @ U_list[i] @ masked_A[i].T for i in range(P)])

    objective = cp.Minimize(0.5 * cp.matrix_frac(y_hot, torch.eye(n) + 2 * preds) + \
                            beta ** 2 * sum([cp.trace(U_list[i]) for i in range(P)]))

    masked_A_2 = [(2 * sign_patterns[i] - 1) * A for i in range(P)]
    constraints = [masked_A_2[i] @ U_list[i] @ masked_A_2[i].T >= 0 for i in range(P)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCS', eps=eps, max_iters=20000, verbose=True)
    solve_time = problem.solver_stats.solve_time
    print('copositive relaxation loss', objective.value)

    return objective.value, solve_time