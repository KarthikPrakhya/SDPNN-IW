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

There are seven experiments ("randomized", "spiral", "iris", "ionosphere", "pima_indians", "bank_notes", "mnist"). The directory `results/no_bias_results` and `results/added_bias_results` will be created with subdirectories `reg_0.1` and `reg_0.01`. Within each subdirectory, 
a subdirectory `{experiment}_results` will be created with a csv file holding the run times and loss values for various 
approaches as well as a figure with the objective loss vs. iteration curves for "randomized" and "spiral" experiments 
and csv files with prediction performance metrics for the rest of the experiments. To run the prediction experiments with bias
added in the first layer, rerun the experiments for "iris", "ionosphere", "pima_indians", "bank_notes" and "mnist"
with the flag "--add_bias" (examples are in the above shell script). Please see our paper for more details:

Prakhya, Karthik, Tolga Birdal, and Alp Yurtsever. "Convex Formulations for Training Two-Layer ReLU Neural Networks." arXiv preprint 
arXiv:2410.22311 (2024).
