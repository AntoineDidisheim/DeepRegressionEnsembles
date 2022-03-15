# DeepRegressionEnsembles
This package proposes an easy application of the paper Deep Regressions Ensemble.

PAPER: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4049493

## Installation
pip install DeepRegressionEnsembles

## Authors
* Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)
* Bryan T. Kelly (Yale SOM; AQR Capital Management, LLC; National Bureau of Economic Research (NBER), bryan.kelly@yale.edu)
* Semyon Malamud (Ecole Polytechnique Federale de Lausanne; Centre for Economic Policy Research (CEPR); Swiss Finance Institute, semyon.malamud@epfl.ch)


## Instruction

See demo.py for a simple example. 

The main class DeepRegressionEnsemble() can be defined with the following parameters: 
* depth: maximum depth
* k: number of ensembles
* p: number of features per ensemble
* lbd: a grid of lambda
* seed: starting TensorFlow seed
* save_dir: specify one to save the model in training (or as a default loading directory)
* perf_measure: 'R2', 'MSE', or 'MAE' --> only define how we print performance
* gamma_min: minimum value of the scaling parameters
* gamma_max: maximum value of the scaling parameter
* gamma_type: 'RANDOM' or 'GRID
* output_layer_dim_reduction: if none, the output layer is as in the paper otherwise, you can specify the dimension to reduce it too
* max_para: the maximum number of independent threads in the TF while loops
* verbose: define whether or not to print progress in training
