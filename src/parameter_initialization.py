import numpy as np

class ParameterInitialization:

    """
    This Class initializes random values for initial weights and biases of our neural network
    """

    def __init__(self, input_size, hidden_layers, n_neurons, n_classes):

        """
        input_size is the number of features or columns
        hidden_layers is the number of hidden layers of the network
        n_neurons is the number of neurons for the hidden layers
        n_classes is the number of class labels
        """

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.n_neurons = n_neurons
        self.n_classes = n_classes

    def initialize_parameters(self):

        weights = {}
        biases = {}

        for i in range(self.hidden_layers):

            if i == 0:

                #For the first hidden layer
                weights[f"W{i + 1}"] = np.random.rand(self.n_neurons, self.input_size) - 0.5
                biases[f"b{i + 1}"] = np.random.rand(self.n_neurons, 1) - 0.5
                
            else:

                # For rest of the hidden layers
                weights[f"W{i + 1}"] = np.random.rand(self.n_neurons, self.n_neurons) - 0.5
                biases[f"b{i + 1}"] = np.random.rand(self.n_neurons, 1) - 0.5

        #For final output layer
        weights[f"W{self.hidden_layers + 1}"] = np.random.rand(self.n_classes, self.n_neurons) - 0.5
        biases[f"b{self.hidden_layers + 1}"] = np.random.rand(self.n_classes, 1) - 0.5

        return weights, biases

