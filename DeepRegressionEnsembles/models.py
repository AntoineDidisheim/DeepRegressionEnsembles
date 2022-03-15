import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class Model(tf.Module):
    def __init__(self,perf_measure = 'MSE'):
        super().__init__()
        self.perf_measure = perf_measure

    def convert_to_numpy(self, v):
        if 'tensor' in str(type(v)):
            v = v.numpy()
        return v

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict(self, x):
        pass

    def performance(self, y_pred, y_true):
        """
        compute the performance
        """
        y_pred = self.convert_to_numpy(y_pred)
        y_true = self.convert_to_numpy(y_true)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        perf = -1
        if self.perf_measure == 'R2':
            ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
            ss_res = np.sum(np.square(y_pred - y_true), axis=0)
            perf = 1 - ss_res / ss_tot
        if self.perf_measure == 'MSE':
            perf = np.mean(np.square(y_pred - y_true), axis=0)
        if self.perf_measure == 'MAE':
            perf = np.mean(np.abs(y_pred - y_true), axis=0)
        if self.verbose:
            tf.print('# validation', self.perf_measure, ',', perf)
        return perf

    def score(self, x, y):
        """
        call prediction and tensor
        :param x:
        :param y:
        :return:
        """
        tf.print('in score')
        pred = self.predict(x)

        tf.print('in performance')
        perf = self.performance(pred, y)
        return perf
    
    def _load_params(self, load_dir):
        p = pd.read_pickle(load_dir + '/model_params.p')
        for k in p.index:
            self.__dict__[k] = p[k]


class RidgeRegressionInside(Model):
    def __init__(self, input_dim, lbd=None, perf_measure='MSE', max_para_across_lambda=100, save_dir=None, verbose = False):
        super().__init__(perf_measure=perf_measure)
        self.perf_measure = perf_measure
        self.verbose =verbose
        self.precision = tf.float32
        self.save_dir = save_dir
        self.debug_print = False
        self.val_perf = []
        self.best_lbd = None
        self.max_para_across_lambda = max_para_across_lambda
        if lbd is None:
            self.lbd = [0.0001, 0.001, 0.01, 0.1, 1.05, 1]
        else:
            self.lbd = lbd
        self.betas = tf.Variable(tf.zeros(shape=(len(self.lbd), input_dim), dtype=self.precision), dtype=self.precision, name=f'betas_Ridge')

    @tf.function
    def train(self, x, y):
        betas = self.estimate_beta_eigen(x, y, signal_output_dim=x.shape[1], large_covariance=False)
        if self.debug_print:
            tf.print('eagerly output', tf.executing_eagerly())
        self.betas.assign(betas)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict(self, x):
        """
        :param x: input before the layer
        :return: output of the output layer
        """
        pred = tf.matmul(x, tf.transpose(self.betas))
        return pred

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def get_next_signals(self, x):
        """
        we include this empty fucntion just to avoid overwritting the save funciton
        """
        return tf.random.normal(shape=(1,1))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def run_output_layer(self, x):
        return tf.random.normal(shape=(1,1))

    def estimate_beta_eigen(self, x, y, signal_output_dim, large_covariance=False):
        """
        get the beta of a ridge regressions given the self.lbd grid of shrinkage parameter.
        We run each beta in a parallel loop (maybe we will ahve to change it, to see)
        :param x:
        :param y:
        :return:
        """
        if self.debug_print:
            tf.print('eagerly estimate beta', tf.executing_eagerly())

        # output_dim = tf.constant(x.shape[1])
        shrinkage_list = tf.convert_to_tensor(self.lbd, dtype=self.precision)

        eigvec1, eigval, data_for_covariance, mu = self.get_eigen_decomposition(x, y, large_covariance)

        if large_covariance:
            eigvec = tf.matmul(data_for_covariance, tf.matmul(eigvec1, tf.linalg.diag((tf.math.abs(eigval) * x.shape[0]) ** (-1 / 2))))
        else:
            eigvec = eigvec1

        multiplied = tf.matmul(tf.transpose(eigvec), mu)
        ## setting up a tf_while loop
        i = tf.constant(0, name='i_in_loop_eig')
        l = tf.zeros(shape=(signal_output_dim, 0), dtype=self.precision, name='output_loop_eig')
        l_index = tf.zeros(shape=(0), dtype=tf.int32)
        cond = lambda i, L, l_index: tf.less(i, shrinkage_list.shape[0])

        def body(i, l, l_index):
            z = tf.gather(shrinkage_list, i)
            m = tf.reshape((1 / (eigval + z)), (-1, 1))
            m = tf.multiply(m, multiplied)
            m = tf.reshape(m, (signal_output_dim, 1))
            l = tf.reshape(l, (signal_output_dim, -1))
            l = tf.concat([l, m], axis=1)
            l_index = tf.concat([l_index, [i]], axis=0)
            return i + 1, l, l_index

        _, normalized, index = tf.while_loop(cond, body, [i, l, l_index],
                                             shape_invariants=[i.get_shape(), tf.TensorShape([signal_output_dim, None]), tf.TensorShape([None])], parallel_iterations=self.max_para_across_lambda)
        # reorder the values in case we had a change of order in paraleliziation
        normalized = tf.gather(normalized, index, axis=1)
        betas = tf.transpose(tf.matmul(eigvec, normalized))
        return betas

    def get_eigen_decomposition(self, x, y, large_covariance):
        """
        1) looks at wether or not we need to invert the matrix (big matrix)
        2) estiamte the covariance matrix and the egien decomposition
        :param x: tensor of signals
        :param y: tensor of labels
        :return: tensors necessary to compute the beta with the "eigen trick"
        """
        if self.debug_print:
            tf.print('eagerly eigen', tf.executing_eagerly())
        # if the we have more features than observation, we invert the covariance matrix to minimize the eign decomposition time
        sh0 = tf.constant(x.shape[0], dtype=self.precision)
        # getting mu
        managed_returns = x * y
        mu = tf.reshape(tf.reduce_mean(managed_returns, axis=0), (-1, 1))
        if large_covariance:
            x = tf.transpose(x)

        covariance = tf.matmul(tf.transpose(x), x) / sh0
        eigval, eigvec1 = tf.linalg.eigh(covariance)
        return eigvec1, eigval, x, mu  # , tf.cast(tf.reduce_sum((ind+1)/(ind+1)),dtype=tf.int32)

    def save(self, save_dir):
        """
        Save the layer.
        :param save_dir: string with the desitonation to save the folder
        """
        os.makedirs(save_dir, exist_ok=True)
        tf.saved_model.save(self, save_dir,
                            signatures={
                                'predict': self.predict,
                                'get_next_signals': self.get_next_signals,
                                'run_output_layer': self.run_output_layer
                            }
                            )

    def load(self, load_dir):
        """
        :param load_dir: directory in whch the tf model is saved
        """
        return tf.saved_model.load(load_dir)

class RidgeRegression(Model):
    def __init__(self, input_dim, lbd=None, max_para_across_lambda=100, perf_measure='MSE', save_dir=None, verbose = False):
        super().__init__(perf_measure=perf_measure)
        self.inside = RidgeRegressionInside(input_dim=input_dim, lbd=lbd, perf_measure=perf_measure, max_para_across_lambda=max_para_across_lambda, save_dir=save_dir, verbose = verbose)
        self.val_perf = None
        self.input_dim = input_dim
        self.best_lbd = None
        self.precision = tf.float32
        self.verbose = verbose
        self.save_dir = save_dir
        self.perf_measure = perf_measure
    def _save_model_params(self):
        """
        we defien here the paramters of the model we need to save to reload it
        """
        t = {
            'best_lbd': self.best_lbd
        }
        t = pd.Series(t)
        t.to_pickle(self.save_dir + '/model_params.p')


    def find_best_hyper_parameter(self):
        """
        select the best performing depth and final penalizaiton parameter in the output layer.
        """
        if self.perf_measure == 'R2':
            best_lbd = np.argmax(self.val_perf[:, 0])
        else:
            best_lbd = np.argmin(self.val_perf[:,0])
        self.best_lbd = best_lbd

    def train(self, x, y, x_val=None, y_val=None):
        """
        Train the simple ridge across all lambda and save the model if a save_dir has been defined.
        If you don't define a validaiton sample, training sample will be used as validation sample
        :param x: numpy or tensor of input
        :param y: numpy or tensor of labels
        :param x_val: numpy or tensor of valdiation input
        :param y_val: numpy or tensor of validation labels
        """
        if 'tensor' not in str(type(x)):
            x = tf.cast(x, dtype=self.precision)
            y = tf.cast(y, dtype=self.precision)
        if x_val is not None:
            if 'tensor' not in str(type(x)):
                x_val = tf.cast(x_val, dtype=self.precision)
                y_val = tf.cast(y_val, dtype=self.precision)

        val_perf_list = []
        if self.verbose:
            tf.print('Start solving regression')
        self.inside.train(x, y)
        if x_val is not None:
            val_pred = self.inside.predict(x_val)
            val_perf = self.inside.performance(y_true=y_val, y_pred=val_pred)
        else:
            val_pred = self.inside.predict(x)
            val_perf = self.inside.performance(y_true=y, y_pred=val_pred)
        val_perf_list.append(tf.reshape(val_perf, (-1, 1)))
        self.val_perf = tf.concat(val_perf_list, axis=1).numpy()
        self.find_best_hyper_parameter()
        if self.save_dir is not None:
            self.inside.save(f'{self.save_dir}/')
            self._save_model_params()

    def predict(self,x):
        assert self.best_lbd is not None, 'Please train first your model'
        pred=self.inside.predict(x).numpy()
        return pred[:,self.best_lbd]

    def load_model(self, load_dir=None):
        """
        load a trained model to the specified load dir, or the default dir defined in __init__
        """
        if load_dir is None:
            load_dir = self.save_dir

        self._load_params(load_dir)
        self.inside = self.inside.load(load_dir)
        return self


    def save(self, save_dir):
        """
        Save the layer.
        :param save_dir: string with the desitonation to save the folder
        """
        os.makedirs(save_dir, exist_ok=True)
        tf.saved_model.save(self, save_dir,
                            signatures={
                                'predict': self.predict,
                                'get_next_signals': self.get_next_signals,
                                'run_output_layer': self.run_output_layer
                            }
                            )


class RidgeEnsemble(RidgeRegressionInside):
    def __init__(self, input_dim, k=25, p=100, lbd=None, output_layer_dim_reduction=None, depth=0, max_para=100, verbose=False, debug_print=False,
                 **kwargs):
        """
        This class captures a single ridge regression.
        We have the inside layer with RF and ridge, and the output layer coded at this level.
        The saving of the tensor graph is also done at this level.
        We stack these class up to creates DRE
        :param input_dim: the second dimension of the X getting into the layer
        :param k: number of ensemble
        :param p: number of feature per ensemble
        :param lbd: grid of shrinkage paramters of the layer
        :param output_layer_dim_reduction: size of the dimension reduciton of the output layer (if None, no reduction)
        :param depth: an id of the current level of depth
        :param max_para: The maximum number of parallel computation across K ensemble computation
        :param verbose: true if we want progress to be printed in the training
        :param debug_print: only for my debugging purposes
        """
        super().__init__(input_dim=1)
        self.p = p
        self.depth = depth
        self.output_layer_dim_reduction = output_layer_dim_reduction
        if lbd is None:
            self.lbd = [0.0001, 0.001, 0.01, 0.1, 1.05, 1]
        else:
            self.lbd = lbd
        self.k = k
        self.debug_print = debug_print
        self.verbose = verbose

        self.precision = tf.float32
        self.init_dim = None
        self.input_dim = input_dim
        self.output_reg_dim = None
        self.output_big_cov = False

        self.act = kwargs['act'] if 'act' in kwargs.keys() else tf.nn.relu
        self.min_gamma = kwargs['min_gamma'] if 'min_gamma' in kwargs.keys() else 0.25
        self.max_gamma = kwargs['max_gamma'] if 'max_gamma' in kwargs.keys() else 1.0
        self.gamma_type = kwargs['gamma_type'] if 'gamma_type' in kwargs.keys() else 'GRID'

        self.max_para = max_para
        self.max_para_across_lambda = 1
        #
        self.rf_weights = tf.Variable(tf.zeros(shape=(self.input_dim, self.p, self.k), dtype=self.precision), dtype=self.precision, name=f'rf_weights_{depth}')
        self.rf_const = tf.Variable(tf.zeros(shape=(self.p, self.k), dtype=self.precision), dtype=self.precision, name=f'rf_const_{depth}')
        self.betas = tf.Variable(tf.zeros(shape=(len(self.lbd), self.p, self.k), dtype=self.precision), dtype=self.precision, name=f'betas_{depth}')

        if self.output_layer_dim_reduction is None:
            self.betas_output = tf.Variable(tf.zeros(shape=(len(self.lbd), len(self.lbd) * self.k), dtype=self.precision), dtype=self.precision, name=f'beta_output_{depth}')
        else:
            self.betas_output = tf.Variable(tf.zeros(shape=(len(self.lbd), self.output_layer_dim_reduction), dtype=self.precision), dtype=self.precision, name=f'beta_output_{depth}')
            self.red_matrix = tf.Variable(tf.random.normal(shape=(len(self.lbd) * self.k, self.output_layer_dim_reduction), dtype=self.precision), dtype=self.precision, name=f'red_matrix_{depth}')

    def expand_random_features(self, x, gamma, init_dim):
        """
        :param x: signals to expand with RF (Tensor)
        :param gamma: gamma to scale the random features
        :param init_dim: dimension of the input, also to scale the rf
        :return: Tensors of expanded random features, as well as the scaled weights and constant term to reproduce it.
        """
        if self.debug_print:
            tf.print('eagerly expand', tf.executing_eagerly())
        w = tf.random.normal(shape=(init_dim, self.p), dtype=self.precision) * gamma
        mult = 0.0001
        b = tf.random.uniform(shape=(1, self.p), minval=-np.pi * mult, maxval=np.pi * mult, dtype=self.precision)
        x = self.act(tf.matmul(x, w) + b)
        # if self.normalize_layers:
        #     x = x / tf.sqrt(tf.reshape(tf.math.reduce_variance(x, axis=0), (1,-1)))
        # if self.demean_layers:
        #     x = x - tf.reshape(tf.math.reduce_mean(x, axis=0), (1,-1))
        return x, w, b

    def get_eigen_decomposition(self, x, y, large_covariance):
        """
        1) looks at wether or not we need to invert the matrix (big matrix)
        2) estiamte the covariance matrix and the egien decomposition
        :param x: tensor of signals
        :param y: tensor of labels
        :return: tensors necessary to compute the beta with the "eigen trick"
        """
        if self.debug_print:
            tf.print('eagerly eigen', tf.executing_eagerly())
        # if the we have more features than observation, we invert the covariance matrix to minimize the eign decomposition time
        sh0 = tf.constant(x.shape[0], dtype=self.precision)
        # getting mu
        managed_returns = x * y
        mu = tf.reshape(tf.reduce_mean(managed_returns, axis=0), (-1, 1))
        if large_covariance:
            x = tf.transpose(x)

        covariance = tf.matmul(tf.transpose(x), x) / sh0
        eigval, eigvec1 = tf.linalg.eigh(covariance)
        return eigvec1, eigval, x, mu  # , tf.cast(tf.reduce_sum((ind+1)/(ind+1)),dtype=tf.int32)

    def train_k_layer(self, x, y, init_dim):
        """
        :param x: tensor of signals
        :param y: tensor of labels
        :param init_dim: a constant we need to scale and reshape some tensor
        :return: Tensor containing the signal for the next layer, tensor with all the rf weights concatenanted,
            tensor with all the rf constant concatenated, tensor with all the betas concatenated across ensemble
        """
        if self.debug_print:
            tf.print('eagerly train k', tf.executing_eagerly())
        if self.verbose:
            tf.print('start training the k models')

        i = tf.constant(0, name='i_in_loop_k')
        l = tf.zeros(shape=(x.shape[0], 0), dtype=self.precision, name='train_k_loop_output_pred')
        w_list = tf.zeros(shape=(init_dim, self.p, 0), dtype=self.precision, name='train_k_loop_output_rf_weights')
        b_list = tf.zeros(shape=(self.p, 0), dtype=self.precision, name='train_k_loop_output_rf_cosnt')
        betas_list = tf.zeros(shape=(len(self.lbd), self.p, 0), dtype=self.precision, name='train_k_loop_output_BETAS')
        l_index = tf.zeros(shape=(0), dtype=tf.int32)
        cond = lambda i, L, W, B, BETAS, l_index: tf.less(i, self.k)

        def body(i, l, w_list, b_list, betas_list, l_index):
            betas, s, w, b = self.expand_and_run(x, y, init_dim, x_output_dim=self.p, k=i)
            pred = tf.matmul(s, tf.transpose(betas))
            pred = tf.reshape(pred, (x.shape[0], -1))
            l = tf.reshape(l, (x.shape[0], -1))
            l = tf.concat([l, pred], axis=1)
            # w
            w = tf.reshape(w, (init_dim, self.p, 1))
            w_list = tf.reshape(w_list, (init_dim, self.p, -1))
            w_list = tf.concat([w_list, w], axis=2)
            # B
            b_list = tf.reshape(b_list, (self.p, -1))
            b_list = tf.concat([b_list, tf.transpose(b)], axis=1)
            # BETAS
            betas = tf.reshape(betas, (len(self.lbd), self.p, 1))
            betas_list = tf.reshape(betas_list, (len(self.lbd), self.p, -1))
            betas_list = tf.concat([betas_list, betas], axis=2)

            l_index = tf.concat([l_index, [i]], axis=0)
            return i + 1, l, w_list, b_list, betas_list, l_index

        _, next_x, rf_weights, rf_const, betas, _ = tf.while_loop(cond, body, [i, l, w_list, b_list, betas_list, l_index],
                                                                  shape_invariants=[
                                                                      i.get_shape(),
                                                                      tf.TensorShape([x.shape[0], None]),  # l
                                                                      tf.TensorShape([init_dim, self.p, None]),  # W
                                                                      tf.TensorShape([self.p, None]),  # B
                                                                      tf.TensorShape([len(self.lbd), self.p, None]),  # BETAS
                                                                      tf.TensorShape([None])
                                                                  ],
                                                                  parallel_iterations=self.max_para)

        return next_x, rf_weights, rf_const, betas

    def expand_and_run(self, x, y, init_dim, x_output_dim, k):
        """
        Draw a gamma, and expand the x with features and run compute the betas (on a single ensemble)
        :param x: tensor of input
        :param y: tensor of labels
        :param init_dim: constant for the scaling of RF
        :param x_output_dim: constant for the reformating of tensors
        :return: tensor of all the betas, tensor with the signals expanded, tensor of rf w, tensor of rf constant.
        """
        if self.debug_print:
            tf.print('eagerly expand and run', tf.executing_eagerly())
        if self.gamma_type == 'RANDOM':
            gamma = tf.random.uniform(shape=(1, 1), minval=self.min_gamma, maxval=self.max_gamma, dtype=self.precision)
        else:
            # define gamma in a grid
            gamma = tf.cast(k, dtype=self.precision) * (self.max_gamma - self.min_gamma) / tf.cast(self.k, dtype=self.precision) + self.min_gamma

        gamma = gamma / tf.math.sqrt(tf.cast(init_dim, dtype=self.precision))
        expanded_x, w, b = self.expand_random_features(x, gamma, init_dim)
        betas = self.estimate_beta_eigen(expanded_x, y, signal_output_dim=x_output_dim)
        return betas, expanded_x, w, b

    def estimate_beta_eigen(self, x, y, signal_output_dim, large_covariance=False):
        """
        get the beta of a ridge regressions given the self.lbd grid of shrinkage parameter.
        We run each beta in a parallel loop (maybe we will ahve to change it, to see)
        :param x:
        :param y:
        :return:
        """
        if self.debug_print:
            tf.print('eagerly estimate beta', tf.executing_eagerly())

        # output_dim = tf.constant(x.shape[1])
        shrinkage_list = tf.convert_to_tensor(self.lbd, dtype=self.precision)

        eigvec1, eigval, data_for_covariance, mu = self.get_eigen_decomposition(x, y, large_covariance)

        if large_covariance:
            eigvec = tf.matmul(data_for_covariance, tf.matmul(eigvec1, tf.linalg.diag((tf.math.abs(eigval) * x.shape[0]) ** (-1 / 2))))
        else:
            eigvec = eigvec1

        multiplied = tf.matmul(tf.transpose(eigvec), mu)
        ## setting up a tf_while loop
        i = tf.constant(0, name='i_in_loop_eig')
        l = tf.zeros(shape=(signal_output_dim, 0), dtype=self.precision, name='output_loop_eig')
        l_index = tf.zeros(shape=(0), dtype=tf.int32)
        cond = lambda i, L, l_index: tf.less(i, shrinkage_list.shape[0])

        def body(i, l, l_index):
            z = tf.gather(shrinkage_list, i)
            m = tf.reshape((1 / (eigval + z)), (-1, 1))
            m = tf.multiply(m, multiplied)
            m = tf.reshape(m, (signal_output_dim, 1))
            l = tf.reshape(l, (signal_output_dim, -1))
            l = tf.concat([l, m], axis=1)
            l_index = tf.concat([l_index, [i]], axis=0)
            return i + 1, l, l_index

        _, normalized, index = tf.while_loop(cond, body, [i, l, l_index],
                                             shape_invariants=[i.get_shape(), tf.TensorShape([signal_output_dim, None]), tf.TensorShape([None])], parallel_iterations=self.max_para_across_lambda)
        # reorder the values in case we had a change of order in paraleliziation
        normalized = tf.gather(normalized, index, axis=1)
        betas = tf.transpose(tf.matmul(eigvec, normalized))
        return betas

    def train_output_layer(self, x, y):
        """
        Get the beta of the output layer
        :param x: tensor with the cocnatenated prediction of the ensembles
        :param y: tensor of labels
        """
        if self.verbose:
            tf.print('in output layer')
        if self.output_layer_dim_reduction is None:
            betas = self.estimate_beta_eigen(x, y, signal_output_dim=self.output_reg_dim, large_covariance=self.output_big_cov)
        else:
            x = tf.matmul(x, self.red_matrix)
            betas = self.estimate_beta_eigen(x, y, signal_output_dim=self.output_layer_dim_reduction, large_covariance=False)

        if self.debug_print:
            tf.print('eagerly output', tf.executing_eagerly())
        self.betas_output.assign(betas)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def run_output_layer(self, x):
        """
        Get the output of the output layer
        :param x: tensor of signals (the concatenated prediciton of the K ensemble)
        :return: tensor of prediction of dimension (number of sample, number of lambda)
        """
        if self.output_layer_dim_reduction is not None:
            x = tf.matmul(x, self.red_matrix)
        betas = self.betas_output
        pred = tf.matmul(x, tf.transpose(betas))
        return pred

    @tf.function
    def train(self, x, y):
        """
        Start of the main computaitonal graph.
        :param x: tensor of input
        :param y: tensor of labels
        :return: tensor that can be used as input for next layers
        """
        if self.debug_print:
            tf.print('Eagerly train layer', tf.executing_eagerly())
        self.output_reg_dim = min(x.shape[0], len(self.lbd) * self.k)
        self.output_big_cov = self.output_reg_dim == x.shape[0]

        self.init_dim = x.shape[1]

        x, rf_weights, rf_const, betas = self.train_k_layer(x, y, x.shape[1])
        self.rf_weights.assign(rf_weights)
        self.rf_const.assign(rf_const)
        self.betas.assign(betas)
        self.train_output_layer(x, y)

        return x

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def get_next_signals(self, x):
        """
        :param x: input before the layer
        :return: concatenated predictions of the K ensemble (can be used as input for next layer)
        """
        rf_weights = tf.unstack(self.rf_weights, axis=2)
        betas = tf.unstack(self.betas, axis=2)
        rf_const = tf.unstack(self.rf_const, axis=1)

        pred = []
        for k in range(self.k):
            s = self.act(tf.matmul(x, rf_weights[k]) + rf_const[k])
            pred.append(tf.matmul(s, tf.transpose(betas[k])))
        s = tf.concat(pred, axis=1)
        return s


class DeepRegressionEnsembles(Model):
    def __init__(self, depth=5, k=500, p=100, lbd=None, seed=12345, save_dir=None, perf_measure='R2',
                 gamma_min=0.25,
                 gamma_max=1.0,
                 gamma_type='GRID',
                 output_layer_dim_reduction=None,
                 max_para=10,
                 act = 'relu',
                 normalize_layers = True,
                 demean_layers = True,
                 verbose=False
                 ):
        """
        main module calling a dre
        :param depth: maximum depth
        :param k: number of ensemble
        :param p: number of features per ensemble
        :param lbd: grid of lambda
        :param seed: starting tensorflow seed
        :param save_dir: specify one to save the model in training (or as a default loading directory)
        :param perf_measure: 'R2', 'MSE', or 'MAE' --> only define how we print performance
        :param gamma_min: minimum value of the scaling paramters
        :param gamma_max: maximum value of the scaling parameter
        :param gamma_type: 'RANDOM' or 'GRID
        :param output_layer_dim_reduction: if none, the output layer is as in the paper, otherwise you can specify the dimension to reduce it too
        :param max_para: the maximum number of independent thread in the tf while loops
        :param verbose: define wether or not to print progress in training
        """
        super().__init__(perf_measure=perf_measure)
        self.normalize_layers = normalize_layers
        self.demean_layers = demean_layers
        self.max_para = max_para
        self.output_layer_dim_reduction = output_layer_dim_reduction
        self.precision = tf.float32
        self.gamma_type = gamma_type
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.perf_measure = perf_measure
        self.verbose = verbose
        self.debug_verbose = False
        tf.random.set_seed(seed)
        self.save_dir = save_dir
        self.p = p
        if lbd is None:
            self.lbd = np.sort([0.0001, 0.001, 0.01] + list(np.arange(0.1, 105.1, 5)) + [1.] + [1000., 2000., 5000., 10000.])
        else:
            self.lbd = lbd
        self.k = k
        self.depth = depth
        self.layers = []
        self.val_perf = []
        self.input_shape = {}

        self.act  = tf.math.cos
        if act =='relu':
            self.act = tf.nn.relu

        self.best_depth = depth
        self.best_lbd = 0
        self.m_ = []
        self.s_ = []

    def _save_model_params(self):
        """
        we defien here the paramters of the model we need to save to reload it
        """
        t = {
            'k': self.k,
            'depth': self.depth,
            'input_shape': self.input_shape,
            'p': self.p,
            'best_depth': self.best_depth,
            'best_lbd': self.best_lbd,
            'demean_layers':self.demean_layers,
            'normalize_layers':self.normalize_layers,
            'gamma_min': self.gamma_min,
            'gamma_max': self.gamma_max,
            'gamma_type': self.gamma_type
        }
        t = pd.Series(t)
        t.to_pickle(self.save_dir + '/model_params.p')

        if self.demean_layers & (self.m_ is not None):
            for i, m_ in enumerate(self.m_):
                s_dir = self.save_dir+f'm/'
                os.makedirs(s_dir,exist_ok=True)
                np.save(s_dir+f'{i}.npy',arr=m_)
        if self.normalize_layers & (self.s_ is not None):
            for i, s_ in enumerate(self.s_):
                s_dir = self.save_dir+f's/'
                os.makedirs(s_dir,exist_ok=True)
                np.save(s_dir+f'{i}.npy',arr=s_)

    def create_default_layer(self, x_shape, depth) -> RidgeEnsemble:
        """
        simple funciton creating a individual layer with all parameters from the model
        :param x_shape: the second dimension of the input
        :param depth: a unique id per level
        :return: a layer to train
        """
        layer = RidgeEnsemble(k=self.k, p=self.p, lbd=self.lbd, input_dim=x_shape, depth=depth, verbose=self.verbose,
                              min_gamma=self.gamma_min, max_gamma=self.gamma_max, gamma_type=self.gamma_type,
                              output_layer_dim_reduction=self.output_layer_dim_reduction,
                              max_para=self.max_para,
                              act=self.act,
                              debug_print=self.debug_verbose)
        return layer

    def train(self, x, y, x_val=None, y_val=None):
        """
        Train the full model and save the model if a save_dir has been defined.
        If you don't define a validaiton sample, training sample will be used as validation sample
        :param x: numpy or tensor of input
        :param y: numpy or tensor of labels
        :param x_val: numpy or tensor of valdiation input
        :param y_val: numpy or tensor of validation labels
        """
        if 'tensor' not in str(type(x)):
            x = tf.cast(x, dtype=self.precision)
            y = tf.cast(y, dtype=self.precision)
        if x_val is not None:
            if 'tensor' not in str(type(x)):
                x_val = tf.cast(x_val, dtype=self.precision)
                y_val = tf.cast(y_val, dtype=self.precision)

        self.input_shape = {}
        val_perf_list = []
        for depth in range(self.depth):
            if self.verbose:
                tf.print('Start training layer', depth)
            layer = self.create_default_layer(x.shape[1], depth)
            self.input_shape[depth] = x.shape[1]
            self.layers.append(layer)

            if self.demean_layers:
                m_ = tf.reshape(tf.reduce_mean(x,axis=0),(1,-1))
                x = x-m_
                self.m_.append(m_.numpy())

            if self.normalize_layers:
                s_ = tf.reshape(tf.math.reduce_std(x,axis=0),(1,-1))
                x = x-s_
                self.s_.append(s_.numpy())

            x = layer.train(x, y)
            if x_val is not None:
                if self.demean_layers:
                    x_val = x_val - m_
                if self.normalize_layers:
                    x_val = x_val/s_
                x_val = layer.get_next_signals(x_val)
                val_pred = layer.run_output_layer(x_val)
                val_perf = self.performance(y_true=y_val, y_pred=val_pred)
            else:
                val_pred = layer.run_output_layer(x)
                val_perf = self.performance(y_true=y, y_pred=val_pred)
            val_perf_list.append(tf.reshape(val_perf, (-1, 1)))
            self.val_perf = tf.concat(val_perf_list, axis=1).numpy()
            self.find_best_hyper_parameter()
            if self.save_dir is not None:
                layer.save(f'{self.save_dir}/{depth}/')
                self._save_model_params()

    def find_best_hyper_parameter(self):
        """
        select the best performing depth and final penalizaiton parameter in the output layer.
        """
        if self.perf_measure == 'R2':
            best_depth = np.argmax(np.max(self.val_perf, 0))
            best_lbd = np.argmax(self.val_perf[:, best_depth])
        else:
            best_depth = np.argmin(np.min(self.val_perf, 0))
            best_lbd = np.argmin(self.val_perf[:, best_depth])
        self.best_depth = best_depth
        self.best_lbd = best_lbd

    def predict(self, x, final_depth=-1, keep_best_lbd=True):
        """
        once trained, does a prediction
        :param x: input for training
        :param final_depth: if not defined, we use the optimal depth
        :param keep_best_lbd: if False, we will return a matrix of prediction (one per lbd)
        :return: predicitons tensor
        """
        assert len(self.layers) > 0, 'Please, train the model first'
        if final_depth == -1:
            final_depth = self.best_depth

        for depth in range(final_depth + 1):
            if self.demean_layers:
                x = x-self.m_[depth]
            if self.normalize_layers:
                x = x/self.s_[depth]


            layer = self.layers[depth]
            x = layer.get_next_signals(x)
        pred = layer.run_output_layer(x).numpy()
        if keep_best_lbd:
            pred = pred[:, self.best_lbd]
        return pred

    def load_model(self, load_dir=None):
        """
        load a trained model to the specified load dir, or the default dir defined in __init__
        """
        if load_dir is None:
            load_dir = self.save_dir

        if self.demean_layers & (self.m_ is not None):
            s_dir = load_dir+f'/m/'
            i_list = np.sort([int(x.split('.npy')[0]) for x in os.listdir(s_dir)])
            self.m_ = []
            for i in i_list:
                m_ = np.load(s_dir+f'{i}.npy')
                self.m_.append(m_)
        if self.normalize_layers & (self.s_ is not None):
            s_dir = load_dir+f'/s/'
            i_list = np.sort([int(x.split('.npy')[0]) for x in os.listdir(s_dir)])
            self.s_ = []
            for i in i_list:
                s_ = np.load(s_dir+f'{i}.npy')
                self.s_.append(s_)

        self._load_params(load_dir)

        self.layers = []
        for depth in range(self.depth):
            layer = self.create_default_layer(self.input_shape[depth], depth)
            layer = layer.load(f'{load_dir}/{depth}/')
            self.layers.append(layer)
        return self
