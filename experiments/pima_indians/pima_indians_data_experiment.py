import os
import copy
import torch
import random
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
import concurrent.futures
from algorithms.cvx.cvx_solver import cvx_solver
from algorithms.sgd.sgd_solver_pytorch import sgd_solver_pytorch
from sklearn.model_selection import train_test_split


def run_sgd(sgd_run_permutation):
    dataset = sgd_run_permutation[0]
    run_number = sgd_run_permutation[1]
    m = sgd_run_permutation[2]
    beta = sgd_run_permutation[3]
    batch_size = sgd_run_permutation[4]
    learning_rate = sgd_run_permutation[5]
    device = sgd_run_permutation[6]
    num_sgd_epochs = sgd_run_permutation[7]
    obj_type = sgd_run_permutation[8]

    # Set the random seed differently for each run
    torch.manual_seed(100 + run_number)
    random.seed(100 + run_number)
    np.random.seed(100 + run_number)

    sgd_losses, train_time, model = sgd_solver_pytorch(dataset['X'], dataset['Y'], m, beta, num_sgd_epochs,
                                                       batch_size, learning_rate, obj_type, device)

    return {'run_number': run_number, 'm': m, 'sgd_model': model, 'sgd_losses': sgd_losses,
            'sgd_train_time': train_time, 'obj_type': obj_type}


def run_cvx(cvx_trial_permutation):
    dataset = cvx_trial_permutation[0]
    deg = cvx_trial_permutation[1]
    beta = cvx_trial_permutation[2]
    solver = cvx_trial_permutation[3]
    obj_type = cvx_trial_permutation[4]

    if deg == 2:
        solver = 'SCS'

    cvx_soln, cvx_obj, cvx_time = cvx_solver(dataset['X'], dataset['Y'], beta, verbose=True, solver=solver, 
                                             obj_type=obj_type, degree_cp_relaxation=deg)
    return {'deg_cp_relaxation': deg, 'cvx_soln': cvx_soln, 'cvx_obj': cvx_obj, 'cvx_time': cvx_time,
            'cvx_solver_type': solver, 'obj_type': obj_type}


def run_pima_indians_data_experiment(run_type, add_bias, regularization_parameter, sgd_learning_rate=1e-6, sgd_num_epochs=3000000,
                             deg_cp_relaxation=0, cvx_solver_type='SCS', device='cpu', num_workers=None):
    """
    The run_pima_indians_data_experiment function runs MOSEK solution of our semidefinite relaxation of our lifted
    formulation of the infinite-width neural network (NN) training problem for the Pima Indians Diabetes Database.
    As baseline approaches, it runs the SGD solution of the same training problem.

    Jack W Smith, James E Everhart, WC Dickson, William C Knowler, and Robert Scott Johannes. Using the adap learning
    algorithm to forecast the onset of diabetes mellitus. In Proceedings of the annual symposium on computer application
    in medical care, pp. 261. American Medical Informatics Association, 1988.

    @type run_type: str
    @param run_type: the type of run to do (e.g. "SGD" or "CVX")
    @type add_bias: str
    @param add_bias: whether or not to add bias term to the first layer
    @type regularization_parameter: float
    @param: regularization_parameter: the regularization parameter for NN training
    @type sgd_learning_rate: float
    @param sgd_learning_rate: the SGD learning rate.
    @type sgd_num_epochs: int
    @param sgd_num_epochs: the number of epochs to run SGD for.
    @type deg_cp_relaxation: int
    @param deg_cp_relaxation: the degree of SoS relaxation for completely positive program.
    @type cvx_solver_type: str
    @param cvx_solver_type: the CVXPY solver to use ("MOSEK or "SCS") for degree 0 relaxation (SCS will be used for higher-order relaxations)
    @type device: str
    @param device: Device to use (ex. cpu for CPU or cuda:x for GPU)
    @type num_workers: int
    @param num_workers: maximum of number of threads or workers
    @rtype: dict
    @return: a nested dictionary with the following keys depending on run_type:
        'sgd_results': a dictionary with the following keys:
                'm': the number of hidden neurons
                'sgd_models': list of PyTorch SGD models for all 5 training runs with SGD 
                'sgd_run_losses': the loss for every SGD iteration for all 5 training runs with SGD
                'sgd_train_times': the training time for SGD for all 5 training runs with SGD
        'num_sgd_epochs': number of epochs to run SGD
        'cvx_results':
            'deg_cp_relaxation': the degree of relaxation in the SoS hierarchy for completely positive cone
            'cvx_soln': the attained solution for CVXPY solver on the semidefinite relaxation of the lifted formulation of the neural network 
                        training problem for the given degree of relaxation
            'cvx_obj': the attained objective value for CVXPY solver on the semidefinite relaxation of the lifted formulation of the neural network
                         training problem for the given degree of relaxation
            'cvx_time': the corresponding time for CVXPY solver (MOSEK for degree 0 and SCS for degree 1 or above) to compute solution

    """
    # Size parameters for the Pima Indians dataset
    num_hidden_neurons = [300]  # number of hidden neurons
    d = 8  # input dimension
    c = 1  # output dimension
    n = 383  # number of datapoints

    # Parameters for NN training
    beta = regularization_parameter
    batch_size = n
    learning_rate = sgd_learning_rate
    num_sgd_epochs = sgd_num_epochs
    num_sgd_runs = 5

    # General parameters
    obj_types = ['L2']

    # Load our csv into a dataframe with column names
    df = pd.read_csv(os.path.join('data', 'pima-indians-diabetes.csv'))
    # extract the last column as the labels
    X = df[df.columns[:-1]].to_numpy(dtype=float)
    labels = df[df.columns[-1]]
    labels = pd.get_dummies(labels)
    Y = 1 * labels.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    train_dataset = {'X': X_train, 'Y': Y_train}
    test_dataset = {'X': X_test, 'Y': Y_test}
    dataset = {'train_dataset': train_dataset, 'test_dataset': test_dataset}
    dataset_path = os.path.join('data', 'pima_indians.pkl')

    # Save the train and test datasets
    with open(dataset_path, 'wb') as handle:
        pkl.dump(dataset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    # Add the bias term
    if add_bias:
        old_dataset = copy.deepcopy(dataset)
        num_rows_train = dataset['train_dataset']['X'].shape[0]
        ones_column_train = np.ones((num_rows_train, 1))
        X_train = np.hstack((ones_column_train, old_dataset['train_dataset']['X']))
        num_rows_test = dataset['test_dataset']['X'].shape[0]
        ones_column_test = np.ones((num_rows_test, 1))
        X_test = np.hstack((ones_column_test, old_dataset['test_dataset']['X']))
        train_dataset = {'X': X_train, 'Y': old_dataset['train_dataset']['Y']}
        test_dataset = {'X': X_test, 'Y': old_dataset['test_dataset']['Y']}
        dataset = {'train_dataset': train_dataset, 'test_dataset': test_dataset}

    # Generate the SGD solution for the Pima Indians dataset
    if run_type == 'SGD':
        num_workers_sgd_cases = num_workers if num_workers else len(num_hidden_neurons)
        runs = range(0, num_sgd_runs)
        sgd_trial_permutations = itertools.product([train_dataset], runs, num_hidden_neurons, [beta], [batch_size],
                                                   [learning_rate], [device], [num_sgd_epochs], obj_types)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_sgd_cases) as executor:
            sgd_results = list(executor.map(run_sgd, sgd_trial_permutations))
        
        baselines = {'sgd_results': sgd_results, 'num_sgd_epochs': num_sgd_epochs}

    # Generate CVX solution for the Pima Indians dataset
    if run_type == 'CVX':
        cvx_trial_permutations = itertools.product([train_dataset], deg_cp_relaxation, [beta],
                                               [cvx_solver_type], obj_types)
        cvx_results = []
        for cvx_trial_permutation in cvx_trial_permutations:
            result = run_cvx(cvx_trial_permutation)
            cvx_results.append(result)
        
        baselines = {'cvx_results': cvx_results}

    return baselines
