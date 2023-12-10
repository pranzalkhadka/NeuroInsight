import numpy as np
import pandas as pd
from src.activation_function import ActivationFunction
from src.one_hot_encoding import OneHotEncoding

activation_function = ActivationFunction()
one_hot_encoding = OneHotEncoding()

df = pd.read_csv("data/train.csv")
data = np.array(df)
m,n = data.shape


class BackPropagation:
    
    """
    This class uses Derivative chain rule to perform back propagation
    """

    def back_propagation(self, parameters, layer_outputs, X, Y):
        m = X.shape[1]
        one_hot_Y = one_hot_encoding.one_hot_encoding(Y)
    
        gradients = {}

        # Backpropagation for the output layer
        layer_index = len(parameters) // 2
        dZ = layer_outputs[f"A{layer_index}"] - one_hot_Y
        gradients[f"dW{layer_index}"] = 1 / m * dZ.dot(layer_outputs[f"A{layer_index - 1}"].T)
        gradients[f"db{layer_index}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        # Backpropagation for hidden layers
        for i in range(layer_index - 1, 0, -1):
            dZ = parameters[f"W{i + 1}"].T.dot(dZ) * activation_function.derivative_ReLU(layer_outputs[f"Z{i}"])
            gradients[f"dW{i}"] = 1 / m * dZ.dot(layer_outputs[f"A{i - 1}"].T)
            gradients[f"db{i}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)


        return gradients
