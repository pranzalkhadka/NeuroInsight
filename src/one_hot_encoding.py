import numpy as np

class OneHotEncoding:
    
    """
    We have our final output as probability distribution over 10 classes
    To calculate the loss, we need to convert the label Y to same format
    So for that , we perform one hot encoding which does something like if label = 3 the, encoded format = (0,0,0,1,0,0,0,0,0,0)
    """

    def one_hot_encoding(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T

        return one_hot_Y