import numpy as np
import pandas as pd
from DeepRegressionEnsembles import Data, DeepRegressionEnsembles

if __name__ == "__main__":

    # generate three batches of simulated single neurone data for demonstraiton
    x, y = Data.generate_single_neurone_data()
    x_train, x_val, x_test = np.array_split(x,3,axis=0)
    y_train, y_val, y_test = np.array_split(y,3,axis=0)



    # we define a model with maximum depth 3, an output layer with reduction trick 500, k=100 ensembles.
    # in order to save the model after training, we specify a saving directory.
    model = DeepRegressionEnsembles(verbose=True, depth=3, output_layer_dim_reduction=500, k=100, save_dir='model_save/')
    # training a model is easy:
    model.train(x_train,y_train,x_val=x_val,y_val=y_val)
    # prediction on a new sample
    y_pred = model.predict(x_test)
    # compute performance
    perf = model.performance(y_true=y_test,y_pred=y_pred)
    print('Out of sample performance:', perf[0])


    # to load a model we just create a DRE object and use the load funciton
    model_to_load = DeepRegressionEnsembles()
    model_to_load=model_to_load.load_model('model_save/')
    # we can also get the out of sample performance directly through the "score" wrapper
    r2_after_load = model_to_load.score(x_train,y_train)
    print('Out of sample performance loaded:', r2_after_load[0])


