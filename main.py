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

epochs = 1000
learning_rate = 0.03


parameter_init = ParameterInitialization(input_size = 784, hidden_layers = 2, n_neurons = 300, n_classes = 10)
parameters = parameter_init.initialize_parameters()
# print(parameters['W1'])

forward_propagation = ForwardPropagation()
back_propagation = BackPropagation()
update_parameters = UpdateParameters()
evaluation = Evaluation()

for i in range(epochs):
    layer_outputs = forward_propagation.forward_propagation(parameters, X_train)
    # print(layer_outputs["A1"])
    last_element_key, last_element_value = list(layer_outputs.items())[-1]
    gradients = back_propagation.back_propagation(parameters, layer_outputs, X_train, Y_train)
    # print(gradients)
    parameters = update_parameters.update_parameters(parameters, gradients, learning_rate)
    # print(parameters['W1'])
    if i % 10 == 0:
        print("Epoch:", i)
        prediction = evaluation.predictions(last_element_value)
        print(f"Accuracy: {evaluation.accuracy(prediction, Y_train)}")
        print("~~~~~~~~~~~~~~~~~")

