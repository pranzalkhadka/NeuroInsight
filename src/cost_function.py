class UpdateParameters:

    """
    This class defines the cost function to update weights and biases of our neural network
    """

    def update_parameters(self, weights, biases, s, alpha):

        p1 = {}
        p2 = {}

        l = len(weights)

        for i in range(1, l + 1):

            p1[f"W{i}"] = weights[f"W{i}"] - alpha * s[f"dW{i}"]
            p2[f"b{i}"] = biases[f"b{i}"] - alpha * s[f"db{i}"]

        return p1, p2