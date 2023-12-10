class UpdateParameters:

    """
    Use the cost function to update weights and biases
    """

    def update_parameters(self, parameters, gradients, learning_rate):
        updated_parameters = {}

        num_layers = len(parameters) // 2

        for i in range(1, num_layers + 1):
            updated_parameters[f"W{i}"] = parameters[f"W{i}"] - learning_rate * gradients[f"dW{i}"]
            updated_parameters[f"b{i}"] = parameters[f"b{i}"] - learning_rate * gradients[f"db{i}"]

        return updated_parameters
