import math
import numpy as np
import cvxpy as cp
import scipy.sparse as sparse


def speye(size):
    """
    The speye function creates a Compressed Sparse Column (CSC) identity matrix of the desired size specified by
    the `size` parameter.

    @type size: int
    @param size: the size of the desired sparse identity matrix along one dimension
    @rtype scipy.sparse.csc_matrix
    @return A sparse array that holds an identity matrix of desired size
    """
    return sparse.csc_matrix(np.eye(size))


def spzeros(shape):
    """
    The spzeros function creates a Compressed Sparse Column (CSC) identity matrix of the desired size specified by
    the `size` parameter.

    @type shape: int or tuple of ints
    @param shape: the size of the desired sparse identity matrix
    @rtype scipy.sparse.csc_matrix
    @return A sparse array that holds a zero matrix of desired size
    """
    return sparse.csc_matrix(np.zeros(shape))


def spones(shape):
    """
    The spzeros function creates a Compressed Sparse Column (CSC) matrix of 1s of the desired size specified by
    the `size` parameter.

    @type shape: int or tuple of ints
    @param shape: the size of the desired sparse matrix
    @rtype scipy.sparse.csc_matrix
    @return A sparse array that holds a zero matrix of desired size
    """
    return sparse.csc_matrix(np.ones(shape))


def cvx_solver(X, Y, beta, verbose, solver='MOSEK', obj_type="L2", scs_max_iters=100000, scs_eps_abs=1e-4,
               degree_cp_relaxation=0):
    """
    The cvx_solver function takes a data matrix X that is n x d and a labels matrix that is n x c as inputs and
    solves the semidefinite relaxation of the completely positive formulation of the infinite-width neural network
    training problem for a two-layer fully-connected ReLU neural network (one hidden layer) using CVXPY's MOSEK or
    SCS solver. The user can specify which solver they want to use.

    @type X: numpy.ndarray
    @param X: the data matrix that is of size n x d
    @type Y: numpy.ndarray
    @param Y: the labels matrix that is of size n x c
    @type beta: float
    @param beta: the regularization parameter for NN training
    @type: bool
    @param verbose: flag whether to print verbose output or not
    @type str
    @param obj_type: the objective loss type ('L2' only)
    @type: str
    @param solver: the solver to use ('SCS' or 'MOSEK')
    @type scs_max_iters: int
    @param scs_max_iters: the maximum number of iterations to run SCS
    @type scs_eps_abs: float
    @param scs_eps_abs: the eps_abs tolerance for SCS
    @rtype: numpy.ndarray
    @param degree_cp_relaxtion: the degree of relaxation as per Lasserre hierarchy for completely positive cone.
    @return: the solution to the lifted problem Lambda (see paper for more details)
    @rtype: float
    @return: solution time for solver
    """
    # Check the arguments
    n = X.shape[0]
    d = X.shape[1]
    c = Y.shape[1]

    if Y.shape[0] != n:
        raise ValueError('X and Y should have the same number of instances (n).')

    # Calculate dimensions
    p = 2 * n + d + c  # dimensionality of LAMBDA
    r = 2 * n

    # Define the operators needed to code the objective and constraints
    CPpart = lambda Lambda: Lambda[0:r, 0:r]

    # Define the matrices required to code the objective and constraints
    Palpha = sparse.hstack([speye(n), spzeros([n, n + d + c])])
    Pbeta = sparse.hstack([spzeros([n, n]), speye(n), spzeros([n, d + c])])
    M = sparse.hstack([-speye(n), speye(n), X, spzeros([n, c])])
    Pv = sparse.hstack([spzeros([c, 2 * n + d]), speye(c)])
    Q = sparse.vstack([sparse.hstack([spzeros([2 * n, 2 * n]), spzeros([2 * n, c + d])]),
                       sparse.hstack([spzeros([c + d, 2 * n]), speye(d + c)])])

    # Define the A operator and its transpose
    # Note: we used trace(...) constraints in the paper but diag(...) gives better performance due to stopping criteria
    # of solvers. Also, M*Lambda*M.T = 0 works better than diag(M*Lambda*M.T) = 0 for same reason.
    Aop = lambda Lambda: cp.hstack([cp.deep_flatten(cp.upper_tri(M @ Lambda @ cp.transpose(M))),
                                    diagAB_cvx(M, Lambda @ cp.transpose(M)),
                                    diagAB_cvx(Palpha, Lambda @ cp.transpose(Pbeta))]).reshape(
        [1, n + math.floor(0.5 * n * (n + 1))])

    # Define the objective function
    if obj_type == "L2":
        obj = lambda Lambda: 0.5 * cp.atoms.power(
            cp.atoms.norm(Palpha @ Lambda @ sparse.csc_matrix.transpose(Pv) - Y, 'fro'),
            2) + 0.5 * beta * cp.sum(cp.multiply(Q, Lambda))
    else:
        return ValueError("Desired loss function is not implemented (obj_type can be L2 only")

    # Construct the problem in CVX
    Lambda = cp.Variable((p, p))
    cost = obj(Lambda)
    objective = cp.Minimize(cost)
    constraints = [Lambda >> 0, Lambda == Lambda.T, Aop(Lambda) == np.zeros([1, n + math.floor(0.5 * n * (n + 1))]),
                   CPpart(Lambda) >= 0]
    if degree_cp_relaxation == 0:
        prob = cp.Problem(objective, constraints)
    else:
        raise ValueError("Deg CP relaxation greater than 0 is not supported.")

    # The optimal objective is returned by prob.solve()
    if solver == 'MOSEK':
        try:
            result = prob.solve(solver=cp.MOSEK, verbose=verbose)
        except cp.error.SolverError:
            result = prob.solve(solver=cp.SCS, verbose=verbose, max_iters=scs_max_iters, eps=scs_eps_abs)
    elif solver == 'SCS':
        result = prob.solve(solver=cp.SCS, verbose=verbose, max_iters=scs_max_iters, eps=scs_eps_abs)
    else:
        raise ValueError('Solver not supported.')
    solve_time = prob.solver_stats.solve_time

    # Print the optimal solution
    print("Optimal Objective with " + solver + ": ", result)
    print("W matrix with " + solver + ": ", Lambda.value)

    return Lambda.value, result, solve_time


def diagAB_cvx(A, B):
    """
    The diagAB_cvx function takes two matrices and computes diag(AB) as a CVXPY expression .

    @type A: Expr
    @param A: a CVXPY expression for first matrix
    @type B: Expr
    @param B: a CVXPY expression for second matrix
    @rtype: Expr
    @return: a CVXPY expression that holds diag(AB)
    """
    return cp.sum(cp.multiply(A, cp.transpose(B)), axis=1)
