# SDPNN-IW
SDP Relaxations for Training ReLU Activation Infinite-Width Neural Networks 

## Installation
To install the prerequisite Python modules, run the following:\
<code>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
</code>

The following command installs the SDPNN code (replace the paths as needed):\
<code>
python3 setup.py develop -s venv/bin -d venv/lib/python3.11/site-packages
</code>

## Running the Experiments

To run the experiments for the paper, run the following command:\
<code>
    sh run_experiments.sh
</code>

There are nine experiments ("randomized", "spiral", "possum", "iris", "ionosphere", "pima_indians", "bank_notes", "mnist",
"cifar10"). The directory `results` will be created with subdirectories `reg_0.1` and `reg_0.01`. Within each subdirectory, 
a subdirectory `{experiment}_results` will be created with a csv file holding the run times and loss values for various 
approaches as well as a figure with the objective loss vs. iteration curves for "randomized", "spiral" and "possum" experiments 
and csv files with prediction performance metrics for the rest of the experiments. To run the prediction experiments with bias
added in the first layer, rerun the experiments for "iris", "ionosphere", "pima_indians", "bank_notes", "mnist" and "cifar10"
with the flag "--add_bias". Please see our paper for more details:

Prakhya, Karthik, Tolga Birdal, and Alp Yurtsever. "Convex Formulations for Training Two-Layer ReLU Neural Networks." arXiv preprint 
arXiv:2410.22311 (2024).
