# Install the SDPNN code (adjust the paths as needed and no need to rerun if already done)
python3 setup.py develop -s venv/bin -d venv/lib/python3.12/site-packages
# PYTHONPATH=$PYTHONPATH:venv/lib/python3.11/site-packages # (Uncomment if needed, in most environments this may not be necesssary)

# Location of the script
SCRIPT_LOC=scripts

# Creates the results directory with required subdirectories
mkdir results
mkdir results/reg_0.1
mkdir results/reg_0.01

############################################################################################################
# Randomized Dataset Runs
############################################################################################################

# Generates the Randomized dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --run_type SGD --run_experiment --sgd_num_epochs 500000 --sgd_learning_rate 1e-5 --results_dir "results/reg_0.1/randomized_multiple_trials_results" --regularization_parameter 0.1 --size_of_randomized_dataset 25 --num_trials_randomized_exp 100 --num_workers 25

# Generates the Randomized dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --run_type SGD --run_experiment --sgd_num_epochs 500000 --sgd_learning_rate 1e-5 --results_dir "results/reg_0.01/randomized_multiple_trials_results" --regularization_parameter 0.01 --size_of_randomized_dataset 25 --num_trials_randomized_exp 100 --num_workers 25

# Generates the Randomized dataset CVX results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --run_experiment --run_type CVX --results_dir "results/reg_0.01/randomized_multiple_trials_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --size_of_randomized_dataset 25 --num_trials_randomized_exp 100 --cvx_solver "MOSEK"

# Generates the Randomized dataset CVX results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --run_experiment --run_type CVX --results_dir "results/reg_0.1/randomized_multiple_trials_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --size_of_randomized_dataset 25 --num_trials_randomized_exp 100 --cvx_solver "MOSEK" --num_workers 5

# Generate a plot for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --results_dir "results/reg_0.1/randomized_multiple_trials_results" --regularization_parameter 0.1 --plot_results --baselines_to_plot "SGD" "CVX"

# Generate a plot for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "randomized" --results_dir "results/reg_0.01/randomized_multiple_trials_results" --regularization_parameter 0.01 --plot_results --baselines_to_plot "SGD" "CVX"

############################################################################################################
# Spiral Dataset Runs
############################################################################################################

# Generates the Spiral dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --run_experiment --run_type SGD --sgd_num_epochs 8000 --sgd_learning_rate 1e-2 --results_dir "results/reg_0.1/spiral_results" --regularization_parameter 0.1 --num_workers 5

# Generates the Spiral dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --run_experiment --run_type SGD --sgd_num_epochs 8000 --sgd_learning_rate 1e-2 --results_dir "results/reg_0.01/spiral_results" --regularization_parameter 0.01 --num_workers 5

# Generates the Spiral dataset CVX results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --run_experiment --run_type CVX --results_dir "results/reg_0.01/spiral_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --cvx_solver "MOSEK"

# Generates the Spiral dataset CVX results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --run_experiment --run_type CVX --results_dir "results/reg_0.1/spiral_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --cvx_solver "MOSEK"

# Generate a plot for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --results_dir "results/reg_0.1/spiral_results" --regularization_parameter 0.1 --plot_results --baselines_to_plot "SGD" "CVX"

# Generate a plot for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "spiral" --results_dir "results/reg_0.01/spiral_results" --regularization_parameter 0.01 --plot_results --baselines_to_plot "SGD" "CVX"

############################################################################################################
# Iris Dataset Runs
############################################################################################################

# Generates the Iris dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "iris" --run_experiment --run_type SGD --sgd_num_epochs 2000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.1/iris_results" --regularization_parameter 0.1 --num_workers 5

# Generates the Iris dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "iris" --run_experiment --run_type SGD --sgd_num_epochs 2000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.01/iris_results" --regularization_parameter 0.01 --num_workers 5

# Generates the Iris dataset CVX results for gamma = 0.01
python3 $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "iris" --run_experiment --run_type CVX --results_dir "results/reg_0.01/iris_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --cvx_solver "SCS"

# Generates the Iris dataset CVX results for gamma = 0.1
python3 $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "iris" --run_experiment --run_type CVX --results_dir "results/reg_0.1/iris_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --cvx_solver "SCS"

############################################################################################################
# Ionosphere Dataset Runs
############################################################################################################

# Generates the Ionosphere dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "ionosphere" --run_experiment --run_type SGD --sgd_num_epochs 2000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.1/ionosphere_results" --regularization_parameter 0.1 --num_workers 5

# Generates the Ionosphere dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "ionosphere" --run_experiment --run_type SGD --sgd_num_epochs 5000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.01/ionosphere_results" --regularization_parameter 0.01 --num_workers 5

# Generates the Ionosphere dataset CVX results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "ionosphere" --run_experiment --run_type CVX --results_dir "results/reg_0.01/ionosphere_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --cvx_solver "SCS"

# Generates the Ionosphere dataset CVX results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "ionosphere" --run_experiment --run_type CVX --results_dir "results/reg_0.1/ionosphere_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --cvx_solver "SCS"

############################################################################################################
# Pima Indians Dataset Runs
############################################################################################################

# Generates the Pima Indians dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "pima_indians" --run_experiment --run_type SGD --sgd_num_epochs 5000000 --sgd_learning_rate 1e-8 --results_dir "results/reg_0.1/pima_indians_results" --regularization_parameter 0.1 --num_workers 5

# Generates the Pima Indians dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "pima_indians" --run_experiment --run_type SGD --sgd_num_epochs 6000000 --sgd_learning_rate 1e-8 --results_dir "results/reg_0.01/pima_indians_results" --regularization_parameter 0.01 --num_workers 5

# Generates the Pima Indians dataset CVX results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "pima_indians" --run_experiment --run_type CVX --results_dir "results/reg_0.01/pima_indians_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --cvx_solver "SCS"

# Generates the Pima Indians dataset CVX results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "pima_indians" --run_experiment --run_type CVX --results_dir "results/reg_0.1/pima_indians_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --cvx_solver "SCS"

############################################################################################################
# Bank Notes Dataset Runs
############################################################################################################

# Generates the Bank Notes dataset SGD results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "bank_notes" --run_experiment --run_type SGD --sgd_num_epochs 5000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.1/bank_notes_results" --regularization_parameter 0.1 --num_workers 5

# Generates the Bank Notes dataset SGD results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "bank_notes" --run_experiment --run_type SGD --sgd_num_epochs 5000000 --sgd_learning_rate 1e-6 --results_dir "results/reg_0.01/bank_notes_results" --regularization_parameter 0.01 --num_workers 5

# Generates the Bank Notes dataset CVX results for gamma = 0.01
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "bank_notes" --run_experiment --run_type CVX --results_dir "results/reg_0.01/bank_notes_results" --regularization_parameter 0.01 --deg_cp_relaxation 0 --cvx_solver "SCS"

# Generates the Bank Notes dataset CVX results for gamma = 0.1
python3 -u $SCRIPT_LOC/run_nn_experiments_semidefinite_relaxation.py --experiment "bank_notes" --run_experiment --run_type CVX --results_dir "results/reg_0.1/bank_notes_results" --regularization_parameter 0.1 --deg_cp_relaxation 0 --cvx_solver "SCS"

############################################################################################################
# Run the predictions
############################################################################################################

python3 $SCRIPT_LOC/run_semidefinite_relaxation_prediction.py
