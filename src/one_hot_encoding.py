import numpy as np

class OneHotEncoding:
    
    """
    This class performs One hot Encoding on our label
    """

    def one_hot_encoding(self, Y):

        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T

        return one_hot_Y