import csv
import os
import torch
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tueplots import bundles
from experiments.randomized.randomized_data_experiment import run_randomized_data_experiment
from experiments.spiral.spiral_data_experiment import run_spiral_data_experiment
from experiments.iris.iris_data_experiment import run_iris_data_experiment
from experiments.ionosphere.ionosphere_data_experiment import run_ionosphere_data_experiment
from experiments.bank_notes.bank_notes_data_experiment import run_bank_notes_data_experiment
from experiments.pima_indians.pima_indians_data_experiment import run_pima_indians_data_experiment

np.random.seed(100)
torch.manual_seed(100)
random.seed(100)


def plot_results(experiment, results_dir, baselines, cvx_solver_type, baselines_to_plot):
    """
    The plot_results function plots the results in graphical and tabular form (CSV). This function is run by setting
    the flag --plot_results. It generates objective value vs iteration loss curves or dashed lines for various
    approaches. It also generates a CSV file with all the final loss values and solution times for various approaches.

    @type experiment: str
    @param experiment: the experiment to run ("randomized", "spiral", "iris" or "ionosphere")
    @type: str
    @param results_dir: the directory where the figures and csv files holding the results will be put.
    @type: dict
    @param baselines: the baselines in dictionary form (see documentation for run_spiral_data_experiment_baselines() as
                        an example)
    @type cvx_solver_type: str
    @param cvx_solver_type: the CVXPY solver to use ("MOSEK or "SCS") for degree 0 relaxation (SCS will be used for
                            higher-order relaxations)
    @type: list
    @param baselines_to_plot: a list of desired approaches to plot.
    @return: None
    """
    # Plotting colors
    clrs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#00ff00', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
            '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2']

    # Import the Matplotlib settings for ICML from tueplots
    plt.rcParams.update(bundles.icml2022())

    # Save an image of the dataset
    
    plt.figure(1)
    if experiment == 'spiral':
        spiral_dataset_path = os.path.join('data', experiment + '_dataset.pkl')
        with open(spiral_dataset_path, 'rb') as handle:
            dataset = pickle.load(handle)
        X, Y = dataset['X'], dataset['Y']
        num_classes = 3
        legend = []
        for i in range(num_classes):
            plt.scatter(X[Y == i, 0], X[Y == i, 1], color=clrs[i])
            legend.append('Class ' + str(i + 1))
        plt.xlabel('Input Feature 1')
        plt.ylabel('Input Feature 2')
        plt.title(experiment.capitalize() + ' Dataset')
        plt.legend(legend, loc='right')
        filename = os.path.join(results_dir, 'spiral_dataset.pdf')
        plt.savefig(filename, format='pdf', bbox_inches="tight", pad_inches=0)

    # Extract the losses and times into Pandas dataframes for easy manipulation and compute other metrics
    if 'sgd_results' in baselines.keys():
        csv_result_file_name = os.path.join(results_dir, experiment + "_sgd_losses.csv")
        raw_sgd_data = pd.DataFrame.from_records(baselines['sgd_results'])
        raw_sgd_data['final_sgd_losses'] = raw_sgd_data['sgd_losses'].apply(lambda x: x[-1])
        grpby = raw_sgd_data.groupby(['m', 'obj_type'])
        run_min_indices = grpby['final_sgd_losses'].idxmin().to_frame('SGD Run Indices').reset_index()
        a = grpby['final_sgd_losses'].min().to_frame('Min SGD Loss').reset_index()
        sgd_results = a.rename(columns={'m': 'Number of Hidden Neurons', 'obj_type': 'Objective Type'})
        sgd_results = sgd_results.round(3)
        sgd_results.to_csv(csv_result_file_name, index=False)

    if 'sgd_trials' in baselines.keys():
        csv_result_file_name = os.path.join(results_dir, experiment + "_sgd_losses.csv")
        raw_sgd_data = pd.DataFrame.from_records(baselines['sgd_trials'])
        raw_sgd_data['final_sgd_losses'] = raw_sgd_data['sgd_losses'].apply(lambda x: x[-1])
        grpby1 = raw_sgd_data.groupby(['trial', 'm', 'obj_type'])
        a = grpby1['final_sgd_losses'].min().to_frame(name='Min SGD Loss')
        run_min_indices = grpby1['final_sgd_losses'].idxmin().to_frame('SGD Run Indices').reset_index()
        grpby2 = a.groupby('m')
        b1 = grpby2['Min SGD Loss'].mean().to_frame('Average Min SGD Loss')
        b2 = grpby2['Min SGD Loss'].std().to_frame('Std Deviation SGD Loss')
        b3 = b1.join(b2).reset_index()
        sgd_trials = b3.rename(columns={'m': 'Number of Hidden Neurons', 'obj_type': 'Objective Type'})
        sgd_trials = sgd_trials.round(3)
        sgd_trials.to_csv(csv_result_file_name, index=False)

    if 'cvx_results' in baselines.keys() or 'cvx_trials' in baselines.keys():
        csv_result_file_name = os.path.join(results_dir, experiment + "_cvx_losses.csv")
        key = 'cvx_results' if 'cvx_results' in baselines.keys() else 'cvx_trials'
        raw_cvx_data = pd.DataFrame.from_records(baselines[key])
        raw_cvx_data['Algorithm'] = 'SDP-NN'
        raw_cvx_data = raw_cvx_data.round(3)
        if 'cvx_results' in baselines.keys():
            cvx_results = raw_cvx_data.filter(["Algorithm", "cvx_time", "obj_type", "cvx_obj"])
            cvx_results = cvx_results.rename(columns={"cvx_time": "Solution Time", "cvx_obj": "Objective Value", "obj_type": "Objective Type"})
            cvx_results = cvx_results.round(3)
            cvx_results.to_csv(csv_result_file_name, index=False)
        else:
            cvx_trials = raw_cvx_data.filter(["trial", "Algorithm", "cvx_time", "obj_type", "cvx_obj"])
            grpby = cvx_trials.groupby(['Algorithm', 'obj_type'])
            a = grpby['cvx_obj'].mean().to_frame(name='Mean Objective Value')
            b = grpby['cvx_obj'].std().to_frame(name='Std Deviation Objective Value')
            c = grpby.agg({'cvx_time': 'mean'})
            c = a.join(b).join(c)
            cvx_trials = c.reset_index().rename(columns={"cvx_time": "Mean Solution Time", "obj_type": "Objective Type"})
            cvx_trials = cvx_trials.round(3)
            cvx_trials.to_csv(csv_result_file_name, index=False)

    # Plot the Objective Value of various approaches
    if 'cvx_results' in baselines.keys():
        for plot_num, obj_type in enumerate(cvx_results['Objective Type'].unique()):
            plt.figure(3 + plot_num)
            k = 0
            legend_labels = []
        
            if 'SGD' in baselines_to_plot:
                num_sgd_epochs = baselines['num_sgd_epochs']
                for m_val in run_min_indices['m'].unique():
                    min_sgd_loss_index = run_min_indices[run_min_indices.m == m_val]['SGD Run Indices'].iloc[0]
                    plt.loglog(np.arange(1, num_sgd_epochs + 1), raw_sgd_data['sgd_losses'][min_sgd_loss_index],
                               color=clrs[k])
                    legend_labels += ['SGD - ' + str(m_val) + ' Hidden Neurons']
                    k += 1

            if 'CVX' in baselines_to_plot:
                for deg in raw_cvx_data['deg_cp_relaxation'].unique():
                    if deg < 2:
                        solver = cvx_solver_type
                    else:
                        solver = 'SCS'

                    cvx_obj = \
                    raw_cvx_data[(raw_cvx_data['deg_cp_relaxation'] == deg)]['cvx_obj'].iloc[0]
                    plt.hlines(cvx_obj, 0, num_sgd_epochs, color=clrs[k], linestyle='dashed')
                    legend_labels += ['SDP-NN']
                    k += 1

            if experiment == 'spiral':
                plt.legend(legend_labels, ncol=2, loc='lower left')
                _, top = plt.ylim()  # return the current ylim
                bottom = 5
                plt.ylim((bottom, top))  # set the ylim to bottom, top

            plt.ylabel('Objective Loss')
            plt.xlabel('Iteration')
            plt.savefig(os.path.join(results_dir, experiment + '_objective_value.pdf'), format='pdf',
                    bbox_inches="tight", pad_inches=0)

    if 'cvx_trials' in baselines.keys():
        plt.figure(3)
        k = 0
        legend_handles = []  # To store plot handles for legend
        legend_labels = []  # To store corresponding labels

        if 'SGD' in baselines_to_plot:
            num_sgd_epochs = baselines['num_sgd_epochs']

            for m_val in run_min_indices['m'].unique():
                indices = run_min_indices[(run_min_indices['m'] == m_val)]['SGD Run Indices']
                losses = np.array([raw_sgd_data['sgd_losses'][i] for i in indices])

                # Calculate mean and standard deviation
                mean_losses = np.mean(losses, axis=0)
                stdplus = np.mean(np.maximum(losses - mean_losses, 0), axis=0)
                stdminus = np.mean(np.maximum(mean_losses - losses, 0), axis=0)

                # Plot mean with ribbon (mean ± std)
                epochs = np.arange(1, num_sgd_epochs + 1)
                line, = plt.loglog(epochs, mean_losses, color=clrs[k], label=f'SGD - {m_val} Hidden Neurons')
                plt.fill_between(epochs, mean_losses - stdminus, mean_losses + stdplus, color=clrs[k],
                                 alpha=0.3)

                legend_handles.append(line)
                legend_labels.append(f'SGD - {m_val} Hidden Neurons')
                k += 1

        if 'CVX' in baselines_to_plot:

            for deg in raw_cvx_data['deg_cp_relaxation'].unique():
                solver = cvx_solver_type if deg < 2 else 'SCS'

                # Get the cvx_obj for each trial
                cvx_objs = raw_cvx_data[(raw_cvx_data['deg_cp_relaxation'] == deg)]['cvx_obj']

                # Calculate mean and standard deviation
                mean_cvx_obj = np.mean(cvx_objs)
                std_cvx_obj = np.std(cvx_objs)

                line_cvx = plt.hlines(mean_cvx_obj, 1, num_sgd_epochs, color=clrs[k],
                                            linestyle='dashed')
                plt.fill_betweenx([mean_cvx_obj - std_cvx_obj, mean_cvx_obj + std_cvx_obj],
                                  1, num_sgd_epochs, color=clrs[k], alpha=0.3)

                legend_handles.append(line_cvx)
                legend_labels.append('SDP-NN')
                # legend_labels.append(f'{deg}-SOS Burer Rel. {solver}')
                k += 1

            _, top = plt.ylim()  # return the current ylim
            bottom = 1e-3
            plt.ylim((bottom, top))  # set the ylim to bottom, top
            plt.legend(legend_handles, legend_labels, ncols=2, loc='lower left')
            plt.ylabel('Objective Loss')
            plt.xlabel('Iteration')
            plt.savefig(os.path.join(results_dir, experiment + '_objective_value.pdf'),
                        format='pdf',
                        bbox_inches="tight", pad_inches=0)


def run_experiment(experiment, run_type, add_bias, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                   deg_cp_relaxation, size_of_randomized_dataset, num_trials_randomized_exp, cvx_solver_type,
                   results_dir, device, num_workers):
    """
    Runs the baselines for a given choice of experiment. This is invoked by running with the flag --run_baselines

    @type experiment: str
    @param experiment: the experiment to run ("randomized", "spiral", "iris", "ionosphere")
    @type run_type: str
    @param run_type: The type of run to do (e.g. "SGD" or "CVX")
    @type add_bias: str
    @param add_bias: whether or not to add bias term to the first layer
    @type regularization_parameter: float
    @param: regularization_parameter: the regularization parameter for NN training.
    @type sgd_learning_rate: float
    @param sgd_learning_rate: the SGD learning rate.
    @type sgd_num_epochs: int
    @param sgd_num_epochs: the number of epochs to run SGD for.
    @type deg_cp_relaxation: int
    @param deg_cp_relaxation: the degree of SoS relaxation for completely positive program.
    @type size_of_randomized_dataset: int
    @param size_of_randomized_dataset: the size of the randomized dataset for the randomized experiment. Ignored for the other experiments.
    @type num_trials_randomized_exp: int
    @param num_trials_randomized_exp: Number of times to run the randomized experiment. Ignored for the other experiments.
    @type deg_cp_relaxation: int
    @param deg_cp_relaxation: the degree of SoS relaxation for completely positive program.
    @type size_of_randomized_dataset: int
    @param size_of_randomized_dataset: the desired number of entries for the randomized dataset
    @type num_trials_randomized_exp: int
    @param num_trials_randomized_exp: the desired number of times to run the randomized dataset experiment.
    @type cvx_solver_type: str
    @param cvx_solver_type: the CVXPY solver to use ("MOSEK or "SCS") for degree 0 SoS relaxation (SCS will be used for higher-order relaxations)
    @type results_dir: str
    @param results_dir: Directory where the figures and csv files holding the results will be put.
    @type device: str
    @param device: the device on which to run PyTorch training (e.g. "cudaX" or "cpu")
    @type num_workers: int
    @param num_workers: maximum of number of threads or workers (only for SGD)
    @rtype: dict
    @return: the baselines in dictionary form (see documentation for run_spiral_data_experiment_baselines() as
                        an example)
    """
    # Run the experiment
    if experiment == 'randomized':
        results = run_randomized_data_experiment(run_type, regularization_parameter, sgd_learning_rate,
                                                 sgd_num_epochs, deg_cp_relaxation, size_of_randomized_dataset,
                                                 num_trials_randomized_exp, cvx_solver_type, device, num_workers)
    elif experiment == 'spiral':
        results = run_spiral_data_experiment(run_type, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                                             deg_cp_relaxation, cvx_solver_type, device, num_workers)
    elif experiment == 'iris':
        results = run_iris_data_experiment(run_type, add_bias, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                                           deg_cp_relaxation, cvx_solver_type, device, num_workers)
    elif experiment == 'ionosphere':
        results = run_ionosphere_data_experiment(run_type, add_bias, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                                                 deg_cp_relaxation, cvx_solver_type, device, num_workers)
    elif experiment == 'pima_indians':
        results = run_pima_indians_data_experiment(run_type, add_bias, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                                           deg_cp_relaxation, cvx_solver_type, device, num_workers)
    elif experiment == 'bank_notes':
        results = run_bank_notes_data_experiment(run_type, add_bias, regularization_parameter, sgd_learning_rate, sgd_num_epochs,
                                                 deg_cp_relaxation, cvx_solver_type, device, num_workers)

    else:
        raise ValueError('Invalid specifier for experiment type.')

    # Save the results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    results_file = os.path.join(results_dir, run_type + '_' + experiment + '_results.pkl')
    with open(results_file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


if __name__ == '__main__':
    """
    This is the main function that can be invoked to run the experiments. To see list of options, one can run this file
    with the option --help. 
    """
    parser = argparse.ArgumentParser(
        prog='SDP Relaxation for Infinite-Width NN Training',
        description='Solves the SDP Relaxation of the 2-layer Infinite-Width Neural Network Training Problem.')

    parser.add_argument('--experiment', action='store', choices=['randomized', 'spiral', 'iris', 'ionosphere', 'bank_notes', 'pima_indians'],
                        help='Choice of experiment to run.', required=True)
    parser.add_argument('--run_experiment', action='store_true', default=False,
                        help='Flag whether to run the experiment or not.')
    parser.add_argument('--run_type', action='store', default='CVX', type=str,
                        choices=['SGD', 'CVX'],
                        help='The type of run to do (e.g. "SGD" or "CVX")')
    parser.add_argument('--add_bias', action='store_true', default=False,
                        help='Flag whether to use bias in first layer for real-life datasets.')
    parser.add_argument('--plot_results', action='store_true', default=False,
                        help='Flag whether to plot the results or not.')
    parser.add_argument('--results_dir', action='store', default='results',
                        help='Directory where the figures and csv files holding the results will be put.')
    parser.add_argument('--baselines_to_plot', type=str, nargs='*',
                        help='Baselines to plot', choices=['SGD', 'CVX'],
                        default=None)
    parser.add_argument('--regularization_parameter', action='store', default=0.1, type=float,
                        help='Regularization parameter for NN training.')
    parser.add_argument('--sgd_learning_rate', action='store', default=1e-2, type=float,
                        help='Initial SGD learning rate.')
    parser.add_argument('--sgd_num_epochs', action='store', default=20000, type=int,
                        help='Number of epochs to run SGD for.')
    parser.add_argument('--deg_cp_relaxation', action='store', default=0, type=int, nargs='+',
                        help='the degree of SoS relaxation for completely positive program.')
    parser.add_argument('--size_of_randomized_dataset', action='store', default=25, type=int,
                        help='the size of the randomized dataset for the randomized experiment. Ignored for the other experiments.')
    parser.add_argument('--num_trials_randomized_exp', action='store', default=1, type=int,
                        help='Number of times to run the randomized experiment. Ignored for the other experiments.')
    parser.add_argument('--cvx_solver_type', action='store', default='SCS', type=str, choices=['MOSEK', 'SCS'],
                        help='The CVXPY solver to use ("MOSEK or "SCS") for degree 0 SoS relaxation (SCS will be used for higher-order relaxations)')
    parser.add_argument('--device', action='store', default='cpu', type=str,
                        help='The device on which to run PyTorch training (e.g. "cudaX" or "cpu")')
    parser.add_argument('--num_workers', action='store', default=1, type=int,
                        help='The maximum of number of threads or workers (only for SGD).')
    args = parser.parse_args()

    # If the user wants to run the baselines, run them or if they have not been run, run them and store them.
    if args.run_experiment:
        baselines = run_experiment(args.experiment, args.run_type, args.add_bias, args.regularization_parameter, args.sgd_learning_rate,
                                   args.sgd_num_epochs, args.deg_cp_relaxation,
                                   args.size_of_randomized_dataset, args.num_trials_randomized_exp,
                                   args.cvx_solver_type, args.results_dir, args.device, args.num_workers)

    # If the user desires to plot the results, plot them.
    if args.plot_results:
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        if os.path.exists(os.path.join(args.results_dir, 'SGD_' + args.experiment + '_results.pkl')) and os.path.exists(os.path.join(args.results_dir, 'CVX_' + args.experiment + '_results.pkl')):
            with open(os.path.join(args.results_dir, 'SGD_' + args.experiment + '_results.pkl'), 'rb') as handle:
                baselines = pickle.load(handle)
            with open(os.path.join(args.results_dir, 'CVX_' + args.experiment + '_results.pkl'), 'rb') as handle:
                cvx_baselines = pickle.load(handle)
            baselines.update(cvx_baselines)
        else:
            baselines = None
        plot_results(args.experiment, args.results_dir, baselines, args.cvx_solver_type, args.baselines_to_plot)
