import numpy as np

from NeuroInsight.activation_function import ActivationFunction
activation_function = ActivationFunction()

from NeuroInsight.one_hot_encoding import OneHotEncoding
onehotencoding = OneHotEncoding()

class BackPropagationForClassification:
    

    def back_propagation_classification(self, a, weights, X_train, Y_train, m):

        one_hot_Y = onehotencoding.one_hot_encoding(Y_train)
        p = {}
        l = len(weights)

        for i in range(l, -1+1, -1):

            if i == l:

                p[f"dZ{i}"] = a[f"A{i}"] - one_hot_Y
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(a[f"A{i-1}"].T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"], axis=1, keepdims=True)

            elif i < l and i > 1:

                p[f"dZ{i}"] = weights[f"W{i+1}"].T.dot(p[f"dZ{i+1}"]) * activation_function.derivative_ReLU(a[f"Z{i}"])
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(a[f"A{i-1}"].T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"], axis=1, keepdims=True)

            elif i == 1:

                p[f"dZ{i}"] = weights[f"W{i+1}"].T.dot(p[f"dZ{i+1}"]) * activation_function.derivative_ReLU(a[f"Z{i}"])
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(X_train.T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"], axis=1, keepdims=True)

        return  p
    


class BackPropagationForRegression:


    def back_propagation_regression(self, a, weights, X_train, Y_train, m):

        p = {}
        l = len(weights)

        for i in range(l, -1+1, -1):

            if i == l:

                p[f"dZ{i}"] = a[f"A{i}"] - Y_train
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(a[f"A{i-1}"].T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"])

            elif i < l and i > 1:

                p[f"dZ{i}"] = weights[f"W{i+1}"].T.dot(p[f"dZ{i+1}"]) * activation_function.derivative_ReLU(a[f"Z{i}"])
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(a[f"A{i-1}"].T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"])

            elif i == 1:
                
                p[f"dZ{i}"] = weights[f"W{i+1}"].T.dot(p[f"dZ{i+1}"]) * activation_function.derivative_ReLU(a[f"Z{i}"])
                p[f"dW{i}"] = 1 / m * p[f"dZ{i}"].dot(X_train.T)
                p[f"db{i}"] = 1 / m * np.sum(p[f"dZ{i}"])

        return  p
