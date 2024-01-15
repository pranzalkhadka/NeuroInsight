import numpy as np

from src.forward_propagation import ForwardPropagationForClassification
forward_propagation_c = ForwardPropagationForClassification()

from src.forward_propagation import ForwardPropagationForRegression
forward_propagation_r = ForwardPropagationForRegression()


class Evaluation:

    """
    This class defines some functions to evaluate the trained model
    """


    def predictions(self, A):
        return np.argmax(A, 0)
    

    def accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    
    def classification_validation_predictions(self, X, weights, biases):
        forward_output = forward_propagation_c.forward_propagation_classification(weights, biases, X)
        key, value = list(forward_output.items())[-1]
        prediction = self.predictions(value)
        return prediction
    

    def regression_validation_predictions(self, weights, biases, X, Y):
        forward_output_val = forward_propagation_r.forward_propagation_regression(weights, biases, X)
        key, value = list(forward_output_val.items())[-1]
        Y_val_predicted = value.flatten()
        rmse_val = self.root_mean_squared_error(Y_val_predicted, Y)
        return rmse_val


    def mean_squared_error(self, Y_predicted, Y):
        return np.mean((Y_predicted - Y)**2)


    def root_mean_squared_error(self, Y_predicted, Y):
        mse = self.mean_squared_error(Y_predicted, Y)
        rmse = np.sqrt(mse)
        return rmse