import numpy as np
import pandas as pd

from src.parameter_initialization import ParameterInitialization
from src.forward_propagation import ForwardPropagation
from src.back_propagation import BackPropagation
from src.cost_function import UpdateParameters
from src.evaluation import Evaluation

df = pd.read_csv("data/train.csv")
data = np.array(df)
m,n = data.shape
np.random.shuffle(data)

val_data = data[0 : 2000].T
Y_val = val_data[0]
X_val = val_data[1 : n]
X_val = X_val / 255.

train_data = data[2000 : m].T
Y_train = train_data[0]
X_train = train_data[1 : n]
X_train = X_train / 255.

epochs = 30
learning_rate = 0.10


parameter_init = ParameterInitialization(input_size = 784, hidden_layers = 2, n_neurons = 300, n_classes = 10)
parameters = parameter_init.initialize_parameters()
print(f"Initial_parameter:{parameters['W1']}")


forward_propagation = ForwardPropagation()
layer_outputs = forward_propagation.forward_propagation(parameters, X_train)
# print(layer_outputs)


back_propagation = BackPropagation()
gradients = back_propagation.back_propagation(parameters, layer_outputs, X_train, Y_train)
# print(gradients)


update_parameters = UpdateParameters()
updated_parameters = update_parameters.update_parameters(parameters, gradients, learning_rate)
print(f"Updated_parameter:{updated_parameters['W1']}")

