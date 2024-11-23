import os
import pickle
import torch
import scipy
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn import metrics
from sklearn.metrics import classification_report
import neural_tangents as nt
from neural_tangents import stax
from jax import jit


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


def create_neural_network_stax(c):
    """
    Creates a neural network using neural_tangents API and returns
    the corresponding kernel function.

    @type c: int
    @param c: the output dimension c
    @return: the kernel function
    """
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1000, W_std=1, b_std=None), stax.Relu(),
        stax.Dense(c, W_std=1, b_std=None)
    )
    kernel_fn = jit(kernel_fn, static_argnames='get')

    return kernel_fn


def get_experiment_name(experiment):
    """
    Returns the full capitalized experiment name.

    @type experiment: str
    @param experiment: short experiment name
    @return: the full experiment name
    """
    return experiment.replace('_', ' ').title()


def get_prediction_accuracy_preds(preds, Y_test):
    """
    Returns the prediction accuracy given the predictions pred and the ground truth Y_test.
    The classification threshold that maximizes the test accuracy is returned.

    @type pred: numpy.ndarray
    @param preds: the predictions
    @type Y_test: numpy ndarray
    @param Y_test: the ground truth (same shape as preds)
    @return: the classification threshold and the test accuracy.
    """
    if preds.shape[0] != Y_test.shape[0] and preds.shape[1] != Y_test.shape[1]:
        raise ValueError('preds and Y_test should have the same dimensions.')
    test_accuracy = 0
    chosen_threshold = 0
    for threshold in np.arange(0, 1, 0.01):
        acc = metrics.accuracy_score(Y_test, preds > threshold)
        if acc > test_accuracy:
            test_accuracy = acc
            chosen_threshold = threshold
    return chosen_threshold, test_accuracy


def run_nngp_and_ntk_prediction(dataset, reg):
    """
    Runs infinite-width neural network prediction with NNGP and NTK kernels. This is equivalent to an infinite ensemble
    of infinite-width networks after marginalizing out the initialization.

    @type dataset: numpy.ndarray
    @param dataset: the dataset containing the train/test split
    @param reg: the value of the regularization parameter.
    @return: the scores, test accuracies and classification thresholds in dictionary form.
    """
    X_train = dataset['train_dataset']['X']
    Y_train = dataset['train_dataset']['Y']
    X_test = dataset['test_dataset']['X']
    Y_test = dataset['test_dataset']['Y']

    # Get the output dimension
    c = Y_test.shape[1]

    # Generate the kernel function
    kernel_fn = create_neural_network_stax(c)

    # Generate the prediction function over an ensemble
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train, diag_reg=reg)

    # Run the prediction with NNGP
    nngp_mean = predict_fn(x_test=X_test, get='nngp', compute_cov=False)
    threshold_nngp, test_accuracy_nngp = get_prediction_accuracy_preds(nngp_mean, Y_test)
    nngp_scores = classification_report(Y_test, nngp_mean > threshold_nngp, output_dict=True)

    # Run the prediction with NTK
    ntk_mean = predict_fn(x_test=X_test, get='ntk', compute_cov=False)
    threshold_ntk, test_accuracy_ntk = get_prediction_accuracy_preds(ntk_mean, Y_test)
    ntk_scores = classification_report(Y_test, ntk_mean > threshold_ntk, output_dict=True)

    return ({'NNGP': nngp_scores, 'NTK': ntk_scores}, {'NNGP': test_accuracy_nngp, 'NTK': test_accuracy_ntk},
            {'NNGP': threshold_nngp, 'NTK': threshold_ntk})


def run_sdp_rounded_prediction(experiment, results_dir, dataset, reg, output_csv_dir):
    """
    Runs the rounding of the solution to the semidefinite relaxation of the completely positive reformulation of the
    infinite-width neural network.

    @type experiment: str
    @param experiment: the short name of the experiment ("iris", "ionosphere", "pima_indians" or "bank_notes")
    @type results_dir: str
    @param results_dir: the location of the .pkl file where the SDP solution is stored
    @type dataset: dict
    @param dataset: a nested dictionary holding the dataset with keys ('train dataset' and 'test_dataset', each with keys
    'X' and 'Y')
    @type reg: float
    @param reg: the value of the regularization parameter
    @type output_csv_dir: str
    @param output_csv_dir: the output directory to store the .pkl files and .csv files with prediction metrics
    @return: four dictionaries holding classification scores, sdp_test_accuracy, sdp_threshold and rounding_metrics with
    rounding metrics meaning the feasibility errors for ReLU constraint (Feas_M) and linear layer constraint (Feas_C).
    """
    X_train = dataset['train_dataset']['X']
    Y_train = dataset['train_dataset']['Y']
    X_test = dataset['test_dataset']['X']
    Y_test = dataset['test_dataset']['Y']

    # Get the input dimension
    n = X_train.shape[0]
    d = X_train.shape[1]
    c = Y_train.shape[1]
    p = 2 * n + d + c

    # Define the objective and prediction functions
    Palpha = sparse.hstack([speye(n), spzeros([n, n + d + c])])
    Pu = sparse.hstack([spzeros([d, 2 * n]), speye(d), spzeros([d, c])])
    Pv = sparse.hstack([spzeros([c, 2 * n + d]), speye(c)])
    Q = sparse.vstack([sparse.hstack([spzeros([2 * n, 2 * n]), spzeros([2 * n, c + d])]),
                       sparse.hstack([spzeros([c + d, 2 * n]), speye(d + c)])])
    obj = lambda Lambda: (
                0.5 * sparse.linalg.norm(Palpha @ sparse.csc_matrix(Lambda) @ Pv.T - sparse.csc_matrix(Y_train)) ** 2
                + 0.5 * reg * sparse.csc_matrix.sum(Q.multiply(sparse.csc_matrix(Lambda))))

    # Load the data
    with open(os.path.join(results_dir, 'CVX_' + experiment + '_results.pkl'), 'rb') as cvx_results:
        data = pickle.load(cvx_results)

    sdp_test_accuracy = {}
    sdp_scores = {}
    sdp_threshold = {}
    rounding_metrics = {}
    device = torch.device('cuda:0')

    n = X_train.shape[0]
    M = np.hstack([-np.eye(n), np.eye(n), X_train, np.zeros([n, c])])

    # Calculate the null space of M
    null_space = scipy.linalg.null_space(M)
    projection_matrix = null_space @ np.linalg.inv(null_space.T @ null_space) @ null_space.T

    for entry in data['cvx_results']:
        alg = 'SDP-NN'
        abbrv_alg = 'sdpnn-iw'
        Lambda_tilde = entry['cvx_soln']
        max_iter = 1000  # 10000
        R = 300  # np.linalg.matrix_rank(Lambda_tilde)
        U, S, Vt = np.linalg.svd(Lambda_tilde, full_matrices=False)
        U = U[:, 0:R]
        S = S[0:R]
        Z = U @ np.diag(np.sqrt(S))
        if Z.shape[1] < 300:
            Z = np.pad(Z, ((0, 0), (0, 300 - Z.shape[1])), 'constant', constant_values=0)
        gamma = 1 / np.linalg.norm(Lambda_tilde, 'fro')
        test_accuracy_sdp_single_alg = 0
        best_preds = np.zeros_like(Y_test)
        best_sdp_threshold = 0
        tos_log_dict = {}
        # Riemannian TOS iteration
        for k in range(max_iter):
            W = projection_P_C(Z, projection_matrix)
            W_tensor = torch.tensor(W, device=device)
            Lambda_tilde_tensor = torch.tensor(Lambda_tilde, device=device)
            V = projection_P_M(2 * W - Z - 2 * gamma * (
                        W_tensor @ (W_tensor.T @ W_tensor) - Lambda_tilde_tensor @ W_tensor).cpu().numpy(), n)
            Z = Z - W + V

            # Calculate the objective
            Lambda = W @ W.T
            lifted_obj = obj(Lambda)
            rounding_obj = 0.5 * np.linalg.norm(Lambda - Lambda_tilde, 'fro') ** 2
            C_feas_error = np.linalg.norm(projection_P_C(W, projection_matrix) - W, 'fro')
            M_feas_error = np.linalg.norm(projection_P_M(W, n) - W, 'fro')

            # Generate predictions with W
            Uweights = Pu @ W
            Vweights = Pv @ W
            temp = X_test @ Uweights
            temp[temp < 0] = 0
            preds = temp @ Vweights.T

            # Print the current test accuracy and objective
            threshold_sdp, test_accuracy = get_prediction_accuracy_preds(preds, Y_test)
            if test_accuracy > test_accuracy_sdp_single_alg:
                test_accuracy_sdp_single_alg = test_accuracy
                best_preds = preds
                best_sdp_threshold = threshold_sdp

            tos_log_dict[k] = {
                'Test Accuracy': round(test_accuracy, 3),
                'Best Test Accuracy': round(test_accuracy_sdp_single_alg, 3),
                'Lifted Obj': round(lifted_obj, 3),
                'Rounding Obj': round(rounding_obj, 3),
                'Feas Error C': round(C_feas_error, 3),
                'Feas Error M': round(M_feas_error, 3),
                'W': W
            }

            print('Iteration:', k, 'Test Accuracy:', round(test_accuracy, 3), 'Best Test Accuracy:',
                  round(test_accuracy_sdp_single_alg, 3), 'Lifted Obj:', round(lifted_obj, 3),
                  'Rounding Obj:', round(rounding_obj, 3), 'Feas Error C:', round(C_feas_error, 3),
                  'Feas Error M:', round(M_feas_error, 3))

        # Save the dictionary to a pickle file
        with open(os.path.join(output_csv_dir, 'tos_log_' + experiment + '_' + str(reg) + '_' + abbrv_alg + '.pkl'),
                  'wb') as f:
            pickle.dump(tos_log_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        sdp_test_accuracy.update({alg: test_accuracy_sdp_single_alg})
        sdp_scores.update({alg: classification_report(Y_test, best_preds > best_sdp_threshold, output_dict=True)})
        sdp_threshold.update({alg: best_sdp_threshold})
        rounding_metrics.update({alg: {'Lifted Obj': lifted_obj, 'Rounding Obj': rounding_obj,
                                       'Feasibility Error C': C_feas_error, 'Feasibility Error M': M_feas_error}})

    return sdp_scores, sdp_test_accuracy, sdp_threshold, rounding_metrics


def projection_P_M(B, n):
    """
    Applies the projection operator P_M to the matrix B for the ReLU constraint.

    @type B: numpy.ndarray
    @param B: The matrix B to be projected (p = 2n + d + c rows, R columns).
    @rtype B_projected: numpy.ndarray
    @return B_projected : the projected matrix
    """
    # Get the dimensions of the matrix B
    p, R = B.shape

    # Create a copy of B to store the projected values
    B_projected = B.copy()

    # Iterate over all rows and columns of B
    for i in range(p):
        for j in range(R):
            if 0 <= i < 2 * n and B[i, j] < 0:
                # Condition 1: B_ij < 0
                B_projected[i, j] = 0
            elif 0 <= i < n and B[i + n, j] > B[i, j]:
                # Condition 2: B_(i+n,j) > B_ij
                B_projected[i, j] = 0
            elif n <= i < 2 * n and B[i - n, j] > B[i, j]:
                # Condition 3: B_(i-n,j) > B_ij
                B_projected[i, j] = 0
            # Otherwise, B_ij remains unchanged (already in B_projected)

    return B_projected


def projection_P_C(B, projection_matrix):
    """
    Projects the vector v using the projection matrix.

    @type B: numpy.ndarray
    @param B : the matrix we are projecting using projection matrix of shape (p, R)
    @type projection_matrix: numpy.ndarray
    @param projection_matrix: the projection matrix for the null space of M of shape (p, p)
    @rtype null_space_projection: numpy.ndarray
    @return null_space_projection: the projection of the vector B onto the null space of M.
    """

    # Initialize the projection matrix
    projection = np.zeros_like(B)

    # Project each column of B onto the null space
    for i in range(B.shape[1]):
        # Project the column B[:, i] onto the null space using projection matrix formed from null_space
        projection[:, i] = projection_matrix @ B[:, i]
    return projection


def run_sgd_prediction(experiment, results_dir, dataset):
    """
    Runs the prediction based on the model learned by Stochastic Gradient Descent (SGD).

    @type experiment: str
    @param experiment: the short name of the experiment ("iris", "ionosphere", "pima_indians" or "bank_notes")
    @type results_dir: str
    @param results_dir: the location of the .pkl file where the SGD solution is stored
    @type dataset: dict
    @param dataset: a nested dictionary holding the dataset with keys ('train dataset' and 'test_dataset', each with keys
    'X' and 'Y')
    @return: three dictionaries holding sgd classification scores, test accuracies and chosen classification thresholds
    """
    X_train = dataset['train_dataset']['X']
    Y_train = dataset['train_dataset']['Y']
    X_test = dataset['test_dataset']['X']
    Y_test = dataset['test_dataset']['Y']

    # Get the input dimension
    n = X_test.shape[0]
    d = X_test.shape[1]

    # Load the data
    with open(os.path.join(results_dir, 'SGD_' + experiment + '_results.pkl'), 'rb') as sgd_results:
        data = pickle.load(sgd_results)

    # Extract the SGD model corresponding to the highest number of hidden neurons (300) and the lowest loss out of 5
    # runs
    raw_sgd_data = pd.DataFrame.from_records(data['sgd_results'])
    raw_sgd_data['final_sgd_losses'] = raw_sgd_data['sgd_losses'].apply(lambda x: x[-1])
    grpby = raw_sgd_data.groupby(['m', 'obj_type'])
    run_min_indices = grpby['final_sgd_losses'].idxmin().to_frame('SGD Run Indices').reset_index()
    min_sgd_loss_index = run_min_indices[run_min_indices.m == 300]['SGD Run Indices'].iloc[0]
    model = raw_sgd_data['sgd_model'][min_sgd_loss_index]
    X_test_torch = torch.tensor(X_test, dtype=torch.float)

    # Evaluate the model
    preds = model(X_test_torch).float()
    test_accuracy_sgd = 0
    threshold_sgd = 0
    for threshold in np.arange(0, 1, 0.01):
        acc = metrics.accuracy_score(Y_test, preds > threshold)
        if acc > test_accuracy_sgd:
            test_accuracy_sgd = acc
            threshold_sgd = threshold

    sgd_test_accuracy = {'SGD - 300 Hidden Neurons': metrics.accuracy_score(Y_test, preds > threshold_sgd)}
    sgd_scores = {'SGD - 300 Hidden Neurons': classification_report(Y_test, preds > threshold_sgd, output_dict=True)}
    sgd_threshold = {'SGD - 300 Hidden Neurons': threshold_sgd}
    return sgd_scores, sgd_test_accuracy, sgd_threshold


def run_prediction(experiment, results_dirs, data_path, regs, output_csv_dir):
    """
    Runs the prediction over all the methods (Rounded SDP, SGD and NNGP/NTK kernel). Calls an auxilliary method
    for each method.

    @type experiment: str
    @param experiment: the short name of the experiment ("iris", "ionosphere", "pima_indians" or "bank_notes")
    @type results_dir: str
    @param results_dir: the location of the .pkl file where the SDP solution is stored
    @type dataset: dict
    @param dataset: a nested dictionary holding the dataset with keys ('train dataset' and 'test_dataset', each with keys
    'X' and 'Y')
    @type regs: list
    @param regs: the values of the regularization parameter to run
    @type output_csv_dir: str
    @param output_csv_dir: the output directory to store the .pkl files and .csv files with prediction metrics
    @return: four dictionaries holding classification scores, test accuracies, chosen classification thresholds and
    rounding metrics with rounding metrics meaning the feasibility errors for ReLU constraint (Feas_M) and linear layer
    constraint (Feas_C) for all the prediction methods.

    """
    # Open the saved test dataset
    with open(data_path, 'rb') as file:
        dataset = pickle.load(file)

    # Runs the prediction
    nngp_ntk_scores_reg = {}
    nngp_ntk_accuracies_reg = {}
    nngp_ntk_thresholds_reg = {}
    # Run the NNGP and NTK based prediction for all values of the regularization parameter
    for reg in regs:
        nngp_ntk_scores, nngp_ntk_accuracies, nngp_ntk_thresholds = run_nngp_and_ntk_prediction(dataset, reg)
        nngp_ntk_scores_reg[reg] = nngp_ntk_scores
        nngp_ntk_accuracies_reg[reg] = nngp_ntk_accuracies
        nngp_ntk_thresholds_reg[reg] = nngp_ntk_thresholds


    rounded_sdp_scores_reg = {}
    rounded_sdp_accuracies_reg = {}
    rounded_sdp_thresholds_reg = {}
    rounded_sdp_rounding_metrics_reg = {}
    # Run the rounded SDP prediction for all values of the regularization parameter
    for results_dir, reg in zip(results_dirs, regs):
        rounded_sdp_scores, rounded_sdp_accuracies, rounded_sdp_thresholds, rounding_metrics = run_sdp_rounded_prediction(
            experiment, results_dir, dataset, reg, output_csv_dir)
        rounded_sdp_scores_reg[reg] = rounded_sdp_scores
        rounded_sdp_accuracies_reg[reg] = rounded_sdp_accuracies
        rounded_sdp_thresholds_reg[reg] = rounded_sdp_thresholds
        rounded_sdp_rounding_metrics_reg[reg] = rounding_metrics

    sgd_scores_reg = {}
    sgd_accuracies_reg = {}
    sgd_thresholds_reg = {}
    # Run the SGD prediction for all values of the regularization parameter
    for results_dir, reg in zip(results_dirs, regs):
        sgd_scores, sgd_accuracies, sgd_thresholds = run_sgd_prediction(experiment, results_dir, dataset)
        sgd_scores_reg[reg] = sgd_scores
        sgd_accuracies_reg[reg] = sgd_accuracies
        sgd_thresholds_reg[reg] = sgd_thresholds

    for reg in regs:
        rounded_sdp_scores_reg[reg].update(nngp_ntk_scores_reg[reg])
        rounded_sdp_scores_reg[reg].update(sgd_scores_reg[reg])
        rounded_sdp_accuracies_reg[reg].update(nngp_ntk_accuracies_reg[reg])
        rounded_sdp_accuracies_reg[reg].update(sgd_accuracies_reg[reg])
        rounded_sdp_thresholds_reg[reg].update(nngp_ntk_thresholds_reg[reg])
        rounded_sdp_thresholds_reg[reg].update(sgd_thresholds_reg[reg])

    return rounded_sdp_scores_reg, rounded_sdp_accuracies_reg, rounded_sdp_thresholds_reg, rounded_sdp_rounding_metrics_reg


if __name__ == '__main__':
    # General Parameters
    output_csv_dir = 'results'
    experiments = ["iris", "ionosphere", "pima_indians", "bank_notes"]
    experiments_dirs = ["iris_results", "ionosphere_results", "pima_indians_results", "bank_notes_results"]
    data_files = ["iris.pkl", "ionosphere.pkl", "pima_indians.pkl", "bank_notes.pkl"]

    # Loop over the experiments and do the prediction and combine them conveniently into Pandas dataframes
    for experiment, experiment_dir, data_file in zip(experiments, experiments_dirs, data_files):
        regs = [0.1, 0.01]
        results_dirs = [os.path.join('results', 'reg_0.1', experiment_dir),
                        os.path.join('results', 'reg_0.01', experiment_dir)]
        data_path = os.path.join('data', data_file)
        sdp_scores, sdp_accuracies, sdp_thresholds, rounding_metrics = run_prediction(experiment, results_dirs,
                                                                                      data_path, regs, output_csv_dir)

        sdp_scores_df = pd.concat(
            {k: pd.concat({a: pd.DataFrame(b) for a, b in v.items()}) for k, v in sdp_scores.items()}).reset_index()
        sdp_scores_df = sdp_scores_df.rename(
            columns={'level_0': 'Regularization Parameter', 'level_1': 'Algorithm', 'level_2': 'Metric Type'}).round(3)
        sdp_accuracies_mod = {'Overall Test Accuracy': sdp_accuracies}
        sdp_accuracies_df = pd.concat(
            {k: pd.DataFrame(v) for k, v in sdp_accuracies_mod.items()}).reset_index()
        sdp_accuracies_df = sdp_accuracies_df.rename(columns={'level_0': 'Metric Type', 'level_1': 'Algorithm'})
        sdp_accuracies_df_1 = sdp_accuracies_df.filter(['Metric Type', 'Algorithm', 0.1])
        sdp_accuracies_df_1['Regularization Parameter'] = 0.1
        sdp_accuracies_df_2 = sdp_accuracies_df.filter(['Metric Type', 'Algorithm', 0.01])
        sdp_accuracies_df_2['Regularization Parameter'] = 0.01
        sdp_accuracies_df_1 = sdp_accuracies_df_1.rename(columns={0.1: 'Metric Value'})
        sdp_accuracies_df_2 = sdp_accuracies_df_2.rename(columns={0.01: 'Metric Value'})
        sdp_accuracies_df = pd.concat([sdp_accuracies_df_1, sdp_accuracies_df_2]).round(3)

        sdp_thresholds_df = pd.DataFrame.from_dict(sdp_thresholds).reset_index().rename(
            columns={'index': 'Algorithm'}).round(2)
        rounding_metrics_df = pd.concat({k: pd.DataFrame(v) for k, v in rounding_metrics.items()}).reset_index().rename(
            columns={'level_0': 'Regularization Parameter', 'level_1': 'Metric Type'}).round(3)
        sdp_final_table_df = sdp_scores_df.filter(
            ["Regularization Parameter", "Algorithm", "Metric Type", "weighted avg"])
        sdp_final_table_df = sdp_final_table_df[sdp_final_table_df["Metric Type"] != "support"]
        sdp_final_table_df['Metric Type'] = ["Weighted Avg. " + value.title() for value in
                                             sdp_final_table_df['Metric Type']]
        sdp_final_table_df = sdp_final_table_df.rename(columns={'weighted avg': "Metric Value"})
        sdp_final_table_df = pd.concat([sdp_final_table_df, sdp_accuracies_df])
        sdp_final_table_df.to_csv(os.path.join(output_csv_dir, experiment + '_sdp_final_results.csv'),
                                  index=False)
        sdp_scores_df.to_csv(os.path.join(output_csv_dir, experiment + '_sdp_raw_classification_scores.csv'),
                             index=False)
        sdp_thresholds_df.to_csv(os.path.join(output_csv_dir, experiment + '_sdp_thresholds.csv'), index=False)
        rounding_metrics_df.to_csv(os.path.join(output_csv_dir, experiment + '_rounding_metrics.csv'), index=False)
        print(sdp_scores_df)
        print(sdp_accuracies_df)
        print(rounding_metrics_df)
