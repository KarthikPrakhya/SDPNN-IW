from setuptools import setup

setup(
    name='SDPNN-IW',
    version='1.0',
    packages=['algorithms', 'data', 'loss_functions', 'models', 'utils', 'experiments'],
    url='https://github.com/KarthikPrakhya/SDPNN-IW',
    license='MIT license',
    author='Karthik Prakhya',
    author_email='karthik.prakhya@umu.se',
    description='Solving Infinite-Width NN Training Problem with Semidefinite Relaxations',
    scripts=['scripts/run_nn_experiments_semidefinite_relaxation.py',
             'scripts/run_semidefinite_relaxation_prediction.py'])
