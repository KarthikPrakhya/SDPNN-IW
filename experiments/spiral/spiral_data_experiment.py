import os
import pickle
import torch
import random
import itertools
import numpy as np
import concurrent.futures
from utils.data.label_encoding import one_hot
from utils.data.datagen_spiral import datagen_spiral
from algorithms.cvx.cvx_solver import cvx_solver
from algorithms.sgd.sgd_solver_pytorch import sgd_solver_pytorch
from algorithms.sahiner.sign_patterns import generate_sign_patterns
from algorithms.sahiner.train_sahiner_frank_wolfe_nn import train_sahiner_frank_wolfe_nn
from algorithms.sahiner.train_sahiner_copositive_relaxation_nn import train_sahiner_copositive_relaxation_nn


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


def run_sahiner(sahiner_trial_permutation):
    dataset = sahiner_trial_permutation[0]
    beta = sahiner_trial_permutation[1]
    fw_epochs = sahiner_trial_permutation[2]
    obj_type = sahiner_trial_permutation[3]
    model = sahiner_trial_permutation[4]

    # Determine the number of sign patterns
    sign_pattern_list, u_vector_list = generate_sign_patterns(dataset['X'], 100000, verbose=True)
    sign_patterns = np.array(sign_pattern_list).astype(np.int32)

    # Run the copositive relaxtion of Sahiner et al.
    loss_sahiner_copositive, time_sahiner_copositive = train_sahiner_copositive_relaxation_nn(dataset['X'],
                                                                                              dataset['Y'],
                                                                                              np.expand_dims(
                                                                                                  sign_patterns, 2),
                                                                                              beta, eps=1e-2)

    # Run the FW algorithm of Sahiner et al.
    t = 0
    for layer, p in enumerate(model.parameters()):
        t += torch.norm(p) ** 2 / 2
    t = t.cpu().item()
    losses_sahiner_fw, times_sahiner_fw = train_sahiner_frank_wolfe_nn(dataset['X'], dataset['Y'],
                                                                       np.expand_dims(sign_patterns, 2),
                                                                       beta, t, epochs=fw_epochs,
                                                                       return_times=True)

    return [{'Algorithm': 'Sahiner FW', 'sahiner_fw_losses': losses_sahiner_fw,
             'sahiner_fw_time': times_sahiner_fw[-1] - times_sahiner_fw[0],
             'obj_type': obj_type},
            {'Algorithm': 'Sahiner Copositive Rel.', 'sahiner_copositive_loss': loss_sahiner_copositive,
             'sahiner_copositive_time': time_sahiner_copositive, 'obj_type': obj_type}]


def run_spiral_data_experiment(run_type, regularization_parameter, sgd_learning_rate=1e-3, sgd_num_epochs=8000,
                               deg_cp_relaxation=0, fw_epochs=15000, sgd_results_file_path=None, cvx_solver_type='MOSEK',
                               device='cpu', num_workers=None):
    """
    The run_spiral_data_experiment function runs MOSEK solution of our semidefinite relaxation of our lifted
    formulation of the infinite-width neural network (NN) training problem for the spiral dataset. As baseline approaches,
    it runs the SGD solution of the same training problem. As a baseline, it runs Sahiner's FW algorithm that solves
    the convex semi-infinite dual formulation as well as Sahiner's copositive relaxation (see paper for more details).

    @type run_type: str
    @param run_type: the type of run ('SGD' or 'CVX')
    @type regularization_parameter: float
    @param: regularization_parameter: the regularization parameter for NN training
    @type sgd_learning_rate: float
    @param sgd_learning_rate: the SGD learning rate.
    @type sgd_num_epochs: int
    @param sgd_num_epochs: the number of epochs to run SGD for.
    @param deg_cp_relaxation: the degree of SoS relaxation for completely positive program.
    @type size_of_randomized_dataset: int
    @type fw_epochs: int
    @param fw_epochs: the number of epochs to run Sahiner's FW algorithm.
    @type sgd_results_file_path: str
    @param sgd_results_file_path: the location of the SGD result file for initializing Sahiner's FW algorithm.
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
    # Size parameters for the spiral dataset
    num_hidden_neurons = [5, 10, 100, 200, 300]  # number of hidden neurons
    d = 2  # input dimension
    c = 3  # output dimension
    n = 60  # number of datapoints

    # Parameters for NN training
    beta = regularization_parameter
    batch_size = n
    learning_rate = sgd_learning_rate
    num_sgd_epochs = sgd_num_epochs
    num_sgd_runs = 5

    # General parameters
    obj_types = ['L2']

    # Load the already created spiral dataset or create it
    ns = 20
    nc = 3
    if os.path.exists(os.path.join('data', 'spiral_dataset.pkl')):
        with open(os.path.join('data', 'spiral_dataset.pkl'), 'rb') as handle:
            loaded_dataset = pickle.load(handle)
        X, Y = loaded_dataset['X'], loaded_dataset['Y']
        if X.shape != (n, d) and Y.shape != (n, c):
            X, Y = datagen_spiral(ns, nc)
    else:
        X, Y = datagen_spiral(ns, nc)
    Y = one_hot(torch.Tensor(Y), nc).numpy()
    dataset = {'X': X, 'Y': Y}

    # Generate the SGD solution for the Spiral dataset
    if run_type == 'SGD':
        num_workers_sgd_cases = num_workers if num_workers else len(num_hidden_neurons)
        runs = range(0, num_sgd_runs)
        sgd_trial_permutations = itertools.product([dataset], runs, num_hidden_neurons, [beta], [batch_size],
                                                   [learning_rate], [device], [num_sgd_epochs], obj_types)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_sgd_cases) as executor:
            sgd_results = list(executor.map(run_sgd, sgd_trial_permutations))

        baselines = {'sgd_results': sgd_results, 'num_sgd_epochs': num_sgd_epochs}

    # Generate CVX solution for the Spiral dataset
    if run_type == 'CVX':
        cvx_trial_permutations = itertools.product([dataset], deg_cp_relaxation, [beta],
                                                   [cvx_solver_type], obj_types)
        cvx_results = []
        for cvx_trial_permutation in cvx_trial_permutations:
            result = run_cvx(cvx_trial_permutation)
            cvx_results.append(result)

        baselines = {'cvx_results': cvx_results}

    # Generate the Sahiner solution for the Spiral dataset
    if run_type == 'Sahiner':
        sahiner_results = []
        if sgd_results_file_path:
            with open(sgd_results_file_path, 'rb') as f:
                sgd_results = pickle.load(f)

            first_models = {}
            for entry in sgd_results['sgd_results']:
                run_number = entry['run_number']
                m = entry['m']
                if run_number == 1 and m == 300:
                    model = entry['sgd_model']
            models = [model]
        else:
            models = []
            sgd_learning_rate = 1e-2
            num_sgd_epochs = 8000
            for obj_type in obj_types:
                _, _, model = sgd_solver_pytorch(dataset['X'], dataset['Y'], 1000, beta, num_sgd_epochs,
                                                               batch_size, sgd_learning_rate, obj_type, device)
                models.append(model)
        sahiner_trial_permutations = itertools.product([dataset], [beta], [fw_epochs], obj_types, models)
        for sahiner_trial_permutation in sahiner_trial_permutations:
            result = run_sahiner(sahiner_trial_permutation)
            sahiner_results += result

        baselines = {'sahiner_results': sahiner_results}

    return baselines
