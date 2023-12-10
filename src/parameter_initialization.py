import numpy as np

class ParameterInitialization:

    """
    This function initializes random values for initial weights and biases of our neural network
    There will be a input layer, two hidden layer with 300 and 100 neurons and output layer with 10 neurons representing mnist digits
    """

    def __init__(self, input_size, hidden_layers, n_neurons, n_classes):
        """
        input_size is the number of features or columns
        n_classes is the number of class labels
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.n_neurons = n_neurons
        self.n_classes = n_classes

    def initialize_parameters(self):

        parameters = {}

        for i in range(self.hidden_layers):

            if i == 0:
                #For the first hidden layer
                parameters[f"W{i + 1}"] = np.random.rand(self.n_neurons, self.input_size) - 0.5
                parameters[f"b{i + 1}"] = np.random.rand(self.n_neurons, 1) - 0.5
            else:
                # For rest of the hidden layers
                parameters[f"W{i + 1}"] = np.random.rand(self.n_neurons, self.n_neurons) - 0.5
                parameters[f"b{i + 1}"] = np.random.rand(self.n_neurons, 1) - 0.5

        #For final output layer
        parameters[f"W{self.hidden_layers + 1}"] = np.random.rand(self.n_classes, self.n_neurons) - 0.5
        parameters[f"b{self.hidden_layers + 1}"] = np.random.rand(self.n_classes, 1) - 0.5

        return parameters

