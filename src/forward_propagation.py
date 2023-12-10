from src.activation_function import ActivationFunction
activation_function = ActivationFunction()

class ForwardPropagation:
    
    """
    This class is responsible for passing our data points through the neural network
    It uses ReLU for hidden layers and softmax for final output layer as activation function
    """


    def forward_propagation(self, parameters, X):
        layer_outputs = {"A0": X}  # Input layer

        for i, (key, value) in enumerate(parameters.items(), start=1):
            if key.startswith("W"):
                # Extract the layer index from the key
                layer_index = int(key[1:])
                W = value
                b = parameters[f"b{layer_index}"]

                Z = W.dot(layer_outputs[f"A{layer_index - 1}"]) + b
                A = activation_function.ReLU(Z) if i < len(parameters) // 2 else activation_function.Softmax(Z)

                layer_outputs[f"Z{layer_index}"] = Z
                layer_outputs[f"A{layer_index}"] = A

        return layer_outputs

