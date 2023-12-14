import numpy as np
from src.forward_propagation import ForwardPropagationForClassification
# from src.forward_propagation import ForwardPropagationForRegression


forward_propagation = ForwardPropagationForClassification()
# forward_propagation = ForwardPropagationForRegression()


class Evaluation:

    """
    This class defines some functions to evaluate the trained model
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes


    def predictions(self, A):
        return np.argmax(A, 0)
    

    def accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    
    def validation_predictions(self, X, weights, biases):
        forward_output = forward_propagation.forward_propagation_classification(weights, biases, X, self.n_classes)
        key, value = list(forward_output.items())[-1]
        prediction = self.predictions(value)
        return prediction
    

    def mean_squared_error(self, Y_predicted, Y):
        Y_predicted = Y_predicted.reshape(Y.shape)
        return np.mean((Y_predicted - Y)**2)