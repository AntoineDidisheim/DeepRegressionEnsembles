# DeepRegressionEnsembles
This package propose an easy application of the paper Deep Regression Ensemble

PAPER: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4049493

## Authors
* Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)
* Bryan T. Kelly (Yale SOM; AQR Capital Management, LLC; National Bureau of Economic Research (NBER), bryan.kelly@yale.edu)
* Semyon Malamud (Ecole Polytechnique Federale de Lausanne; Centre for Economic Policy Research (CEPR); Swiss Finance Institute, semyon.malamud@epfl.ch)


## Instruction

See demo.py for simple example. 

The main class DeepRegressionEnsemble() can be defined with the following parameters: 
* depth: maximum depth
* k: number of ensemble
* p: number of features per ensemble
* lbd: grid of lambda
* seed: starting tensorflow seed
* save_dir: specify one to save the model in training (or as a default loading directory)
* perf_measure: 'R2', 'MSE', or 'MAE' --> only define how we print performance
* gamma_min: minimum value of the scaling paramters
* gamma_max: maximum value of the scaling parameter
* gamma_type: 'RANDOM' or 'GRID
* output_layer_dim_reduction: if none, the output layer is as in the paper, otherwise you can specify the dimension to reduce it too
* max_para: the maximum number of independent thread in the tf while loops
* verbose: define wether or not to print progress in training