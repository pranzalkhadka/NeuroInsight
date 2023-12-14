from src.activation_function import ActivationFunction

activation_function = ActivationFunction()

class ForwardPropagation:
    
    """
    This class is responsible for passing our data points through the neural network
    It uses ReLU for hidden layers and softmax for final output layer as activation function
    """

    def forward_propagation(self, weights, biases, X, n_classes):

        l = len(weights)
        p = {}

        for i in range(1, l+1):

            if i == 1 :

                p[f"Z{i}"] = weights['W1'].dot(X) + biases['b1']
                p[f"A{i}"] = activation_function.ReLU( weights['W1'].dot(X) + biases['b1'])

            elif i > 1 and i < l:

                p[f"Z{i}"] = weights[f"W{i}"].dot(p[f"A{i-1}"]) + biases[f"b{i}"]
                p[f"A{i}"] = activation_function.ReLU(weights[f"W{i}"].dot(p[f"A{i-1}"]) + biases[f"b{i}"])

            elif i == l:
                
                p[f"Z{i}"] = weights[f"W{i}"].dot(p[f"A{i-1}"]) + biases[f"b{i}"]
                if n_classes == 1:
                    p[f"A{i}"] = activation_function.Linear(weights[f"W{i}"].dot(p[f"A{i-1}"]) + biases[f"b{i}"])
                else:
                    p[f"A{i}"] = activation_function.Softmax(weights[f"W{i}"].dot(p[f"A{i-1}"]) + biases[f"b{i}"])

        return p