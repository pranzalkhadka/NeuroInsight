import numpy as np

class ActivationFunction:
    """
    This class defines the activation functions to be used in the neural network
    """

    def ReLU(self, Z):

        """
        For each element in Z, the ReLU function returns Z if Z is positive and return 0 if Z is negative
        """

        return np.maximum(Z ,0)
    
    
    def Softmax(self, Z):

        """
        The Softmax function converts the final score into probability distribution over different classes
        np.exp(Z) calculates the element-wise exponential of input Z
        np.sum(np.exp(Z)) calculates the sum of the exponentiated values of ell elements of Z
        """

        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
        # Z = np.float128(Z)
        # A = np.exp(Z) / sum(np.exp(Z))
        # return A


    def Linear(self, Z):
        """
        Linear activation function for regression problem
        """
        return Z
    

    def derivative_ReLU(self, Z):

        """
        This derivative is used to compute the gradients of the loss with respect to the input of the ReLU activation function to optimize our parameters
        If any element in Z is greater than 0 , return 1 otherwise 0
        """

        # return Z > 0
        return np.where(Z > 0, 1, 0)

    
    def derivative_linear(self, Z):

        return 1

    
