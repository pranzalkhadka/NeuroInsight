import numpy as np


class ParameterInitialization:

    """
    This Class initializes random values for initial weights and biases of our neural network
    """

    def __init__(self, input_size, n_layers, n_neurons_per_layer, n_classes):

        """
        input_size is the number of features or columns
        hidden_layers is the number of hidden layers of the network
        n_neurons is the number of neurons for the hidden layers
        n_classes is the number of class labels
        """
        
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_classes = n_classes

    def initialize_parameters(self):

        weights = {}
        biases = {}

        for i in range(self.n_layers):

            if i == 0:

                weights[f"W{i + 1}"] = np.random.rand(self.n_neurons_per_layer[i], self.input_size) - 0.5
                biases[f"b{i + 1}"] = np.random.rand(self.n_neurons_per_layer[i], 1) - 0.5

            else:
                
                weights[f"W{i + 1}"] = np.random.rand(self.n_neurons_per_layer[i], self.n_neurons_per_layer[i-1]) - 0.5
                biases[f"b{i + 1}"] = np.random.rand(self.n_neurons_per_layer[i], 1) - 0.5

        weights[f"W{self.n_layers + 1}"] = np.random.rand(self.n_classes, self.n_neurons_per_layer[-1]) - 0.5
        biases[f"b{self.n_layers + 1}"] = np.random.rand(self.n_classes, 1) - 0.5

        return weights, biases
