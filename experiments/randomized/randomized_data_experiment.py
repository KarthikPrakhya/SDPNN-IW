import os
import random
import pickle
import torch
import itertools
import numpy as np
import pandas as pd
import concurrent.futures
from models.FCNetwork import FCNetwork
from utils.data.datagen_randomized import datagen_randomized
from algorithms.cvx.cvx_solver import cvx_solver
from algorithms.sgd.sgd_solver_pytorch import sgd_solver_pytorch
from algorithms.sahiner.train_sahiner_frank_wolfe_nn import train_sahiner_frank_wolfe_nn
from algorithms.sahiner.sign_patterns import generate_sign_patterns


def run_sgd(sgd_trial_permutation):
    trial = sgd_trial_permutation[0][0]
    dataset = sgd_trial_permutation[0][1]
    run_number = sgd_trial_permutation[1]
    m = sgd_trial_permutation[2]
    beta = sgd_trial_permutation[3]
    batch_size = sgd_trial_permutation[4]
    learning_rate = sgd_trial_permutation[5]
    device = sgd_trial_permutation[6]
    num_sgd_epochs = sgd_trial_permutation[7]
    obj_type = sgd_trial_permutation[8]

    # Set the random seed differently for each run
    torch.manual_seed(100 + run_number)
    random.seed(100 + run_number)
    np.random.seed(100 + run_number)

    sgd_losses, train_time, model = sgd_solver_pytorch(dataset['X'], dataset['Y'], m, beta, num_sgd_epochs,
                                                       batch_size, learning_rate, obj_type, device)

    return {'trial': trial, 'run_number': run_number, 'm': m, 'sgd_model': model, 'sgd_losses': sgd_losses,
            'sgd_train_time': train_time, 'num_sgd_epochs': num_sgd_epochs, 'obj_type': obj_type}


def run_cvx(cvx_trial_permutation):
    trial = cvx_trial_permutation[0][0]
    dataset = cvx_trial_permutation[0][1]
    deg = cvx_trial_permutation[1]
    beta = cvx_trial_permutation[2]
    solver = cvx_trial_permutation[3]
    obj_type = cvx_trial_permutation[4]

    if deg == 2:
        solver = 'SCS'

    cvx_soln, cvx_obj, cvx_time = cvx_solver(dataset['X'], dataset['Y'], beta, verbose=True, solver=solver,
                                             obj_type=obj_type, degree_cp_relaxation=deg)
    return {'trial': trial, 'deg_cp_relaxation': deg, 'cvx_soln': cvx_soln, 'cvx_obj': cvx_obj, 'cvx_time': cvx_time,
            'cvx_solver_type': solver, 'obj_type': obj_type}


def run_sahiner(sahiner_trial_permutation):
    trial = sahiner_trial_permutation[0][0][0]
    dataset = sahiner_trial_permutation[0][0][1]
    model = sahiner_trial_permutation[0][1]
    beta = sahiner_trial_permutation[1]
    fw_epochs = sahiner_trial_permutation[2]
    obj_type = sahiner_trial_permutation[3]

    # Calculate the Frank-Wolfe solution (Algorithm 1) of the convex semi-infinite dual formulation (Eq. 15) in
    # Sahiner et al. paper
    sign_pattern_list, u_vector_list = generate_sign_patterns(dataset['X'], 100000, verbose=True)
    sign_patterns_int_array = np.array(sign_pattern_list).astype(np.int32)
    t = 0
    for layer, p in enumerate(model.parameters()):
        t += torch.norm(p) ** 2 / 2
    t = t.cpu().item()
    losses_sahiner_fw, times_sahiner_fw = train_sahiner_frank_wolfe_nn(dataset['X'], dataset['Y'],
                                                                       np.expand_dims(sign_patterns_int_array, 2),
                                                                       beta, t, epochs=fw_epochs, use_cvxpy=False,
                                                                       print_freq=200,
                                                                       return_times=True)

    return {'Algorithm': 'Sahiner FW', 'trial': trial, 'sahiner_fw_losses': losses_sahiner_fw,
            'sahiner_fw_time': times_sahiner_fw[-1] - times_sahiner_fw[0], 'obj_type': obj_type}


def run_randomized_data_experiment(run_type, regularization_parameter, sgd_learning_rate=1e-5, sgd_num_epochs=500000,
                                   deg_cp_relaxation=1, size_of_randomized_dataset=25,
                                   num_trials_randomized_exp=10, fw_epochs=50000, sgd_results_file_path=None,
                                   cvx_solver_type='MOSEK', device='cpu', num_workers=None):
    """
    The run_randomized_data_experiment function runs MOSEK solution of our semidefinite relaxation of our lifted
    formulation of the infinite-width neural network (NN) training problem for the randomized dataset. As baseline
    approaches, it runs the SGD solution of the same training problem. As a baseline, it runs Sahiner's FW algorithm
    that solves the convex semi-infinite dual formulation (see paper for more details).

    @type run_type: str
    @param run_type: the type of run to do (e.g. "SGD" or "CVX")
    @type regularization_parameter: float
    @param: regularization_parameter: the regularization parameter for NN training
    @type sgd_learning_rate: float
    @param sgd_learning_rate: the SGD learning rate.
    @type sgd_num_epochs: int
    @param sgd_num_epochs: the number of epochs to run SGD for.
    @type deg_cp_relaxation: int
    @param deg_cp_relaxation: the degree of SoS relaxation for completely positive program.
    @type size_of_randomized_dataset: int
    @param size_of_randomized_dataset: the desired number of entries for the randomized dataset
    @type num_trials_randomized_exp: int
    @param num_trials_randomized_exp: the desired number of times to run the randomized dataset experiment.
    @type fw_epochs: int
    @param fw_epochs: the number of epochs to run Sahiner's FW algorithm.
    @type sgd_result_file_path: str
    @param sgd_result_file_path: the location of the SGD result file for initializing Sahiner's FW algorithm.
    @type cvx_solver_type: str
    @param cvx_solver_type: the CVXPY solver to use ("MOSEK or "SCS") for degree 0 relaxation (SCS will be used for higher-order relaxations)
    @type device: str
    @param device: the device on which to run PyTorch training (e.g. "cudaX" or "cpu")
    @type num_workers: int
    @param num_workers: maximum of number of threads or workers
    @type: dict
    @return: a nested dictionary with the following keys:
        'sgd_trials':
            'sgd_results': a dictionary with the following keys:
                'm': the number of hidden neurons
                'sgd_models': list of PyTorch SGD models for all 5 training runs with SGD 
                'sgd_run_losses': the loss for every SGD iteration for all 5 training runs with SGD
                'sgd_train_times': the training time for SGD for all 5 training runs with SGD
        'num_sgd_epochs': number of epochs to run SGD
        'cvx_trials':
            'cvx_results':
                'deg_cp_relaxation': the degree of relaxation in the SoS hierarchy for completely positive cone
                'cvx_soln': the attained solution for CVXPY solver on the semidefinite relaxation of the lifted formulation of the neural network 
                            training problem for the given degree of relaxation
                'cvx_obj': the attained objective value for CVXPY solver on the semidefinite relaxation of the lifted formulation of the neural network
                            training problem for the given degree of relaxation
                'cvx_time': the corresponding time for CVXPY solver (MOSEK for degree 0 and SCS for degree 1 or above) to compute solution
    """
    # Size parameters for the randomized dataset
    num_hidden_neurons = [5, 10, 100, 200, 300]  # number of hidden neurons
    m_gen = 100  # number of hidden neurons in generator network for dataset
    d = 2  # input dimension
    c = 5  # output dimension
    n = size_of_randomized_dataset  # 25  # number of datapoints

    # General parameters
    obj_types = ['L2']

    # Name of the pickle file to store the randomized dataset
    dataset_file_name = os.path.join('data', 'randomized_datasets_n_' + str(n) + '_d_' + str(d) + '_c_' + str(
        c) + '_num_datasets_' + str(num_trials_randomized_exp) + '.pkl')

    # Parameters for NN training
    beta = regularization_parameter
    batch_size = n
    learning_rate = sgd_learning_rate
    num_sgd_epochs = sgd_num_epochs
    num_sgd_runs = 5

    # Load the already created randomized dataset or create it
    if os.path.exists(dataset_file_name):
        with open(dataset_file_name, 'rb') as handle:
            datasets = pickle.load(handle)
    else:
        datasets = []
        for i in range(num_trials_randomized_exp):
            random.seed(i + 100)
            X, Y = datagen_randomized(m_gen, d, c, n)
            datasets.append({'X': X, 'Y': Y})
        with open(dataset_file_name, 'wb') as handle:
            pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create a pairing of trials and datasets
    trials = range(0, num_trials_randomized_exp)
    dataset_trials = zip(trials, datasets)

    # Generate the SGD solution for the Randomized dataset
    if run_type == 'SGD':
        num_workers_sgd_trials = num_workers if num_workers else len(num_hidden_neurons)
        runs = range(0, num_sgd_runs)
        sgd_trial_permutations = itertools.product(dataset_trials, runs, num_hidden_neurons, [beta], [batch_size],
                                                   [learning_rate], [device], [num_sgd_epochs], obj_types)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_sgd_trials) as executor:
            sgd_trials = list(executor.map(run_sgd, sgd_trial_permutations))

        baselines = {'sgd_trials': sgd_trials, 'num_sgd_epochs': num_sgd_epochs}

    # Generate CVX solution for the Randomized dataset
    if run_type == 'CVX':
        cvx_trial_permutations = itertools.product(dataset_trials, deg_cp_relaxation, [beta],
                                                   [cvx_solver_type], obj_types)
        cvx_trials = []
        for cvx_trial_permutation in cvx_trial_permutations:
            result = run_cvx(cvx_trial_permutation)
            cvx_trials.append(result)

        baselines = {'cvx_trials': cvx_trials}

    # Generate the Sahiner solution for the Randomized dataset
    if run_type == 'Sahiner':
        sahiner_trials = []
        if sgd_results_file_path:
            with open(sgd_results_file_path, 'rb') as f:
                sgd_results = pickle.load(f)

            first_models = {}
            for entry in sgd_results['sgd_trials']:
                trial = entry['trial']
                run_number = entry['run_number']
                m = entry['m']
                if trial not in first_models and run_number == 1 and m == 300:
                    first_models[trial] = entry['sgd_model']
            models = [value for key, value in sorted(first_models.items())]
        else:
            models = []
            sgd_learning_rate = 1e-5
            num_sgd_epochs = 500000
            for dataset in datasets:
                _, _, model = sgd_solver_pytorch(dataset['X'], dataset['Y'], 1000, beta, num_sgd_epochs,
                                                               batch_size, sgd_learning_rate, obj_types[0], device)
                models.append(model)
        dataset_trials_models = zip(dataset_trials, models)
        sahiner_trial_permutations = itertools.product(dataset_trials_models, [beta], [fw_epochs], obj_types)
        for sahiner_trial_permutation in sahiner_trial_permutations:
            result = run_sahiner(sahiner_trial_permutation)
            sahiner_trials.append(result)

        baselines = {'sahiner_trials': sahiner_trials}

    return baselines
