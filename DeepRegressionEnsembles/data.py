import numpy as np


class Data:  # a simple class to fascilatate loading of the data
    @staticmethod
    def generate_single_neurone_data(
            noise_index=1,
            full_sample_size=1000,
            intrinsic_dimension=50,
            seed_=0,
            nb_neurones=1,
            act='relu'
    ):
        np.random.seed(seed_)
        raw_features = np.random.normal(size=(full_sample_size, intrinsic_dimension))
        weight_matrix = np.random.normal(size=(intrinsic_dimension, nb_neurones)).clip(-3, 3)
        std_ = np.linspace(0, 1, 10)[noise_index]
        noise = np.random.normal(scale=std_, size=(full_sample_size, 1))
        if nb_neurones == 1:
            labels = raw_features @ weight_matrix
            if act == 'sigmoid':
                labels = 1 / (1 + np.exp(-labels))
            else:
                labels = np.maximum(labels, 0)
                labels = labels / np.std(labels)
        else:
            linear_coefficients_for_combining_neurons = np.random.normal(size=(nb_neurones, 1)).clip(-3, 3)
            labels = raw_features @ weight_matrix
            if act == 'sigmoid':
                labels = 1 / (1 + np.exp(-labels))
                labels = labels @ linear_coefficients_for_combining_neurons
            else:
                labels = np.maximum(labels, 0)
                labels = labels @ linear_coefficients_for_combining_neurons
                labels = labels / np.std(labels)

        labels = labels + noise
        return raw_features, labels

