import numpy as np
from src.forward_propagation import ForwardPropagation

forward_propagation = ForwardPropagation()

class Evaluation:
    """
    This class defines some functions to evaluate the trained model
    """

    def predictions(self, A):
        return np.argmax(A, axis = 0)
    

    def accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    