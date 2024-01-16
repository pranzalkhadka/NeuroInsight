import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Visualization:


    def plot_loss_curve(self, losses):
        plt.plot(losses)
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()



    def plot_accuracy_curve(self, accuracies):
        plt.plot(accuracies)
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    

    def plot_neural_network_architecture(self, n_input, n_layers, n_neurons_per_layer, n_output):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=[0.5] * n_input, y=[i for i in range(n_input)], mode='markers', marker=dict(size=30),
                                text=[f'Feature {i}' for i in range(1, n_input + 1)], name='Input Layer'))

        for i, neurons in enumerate(n_neurons_per_layer, start=1):
            fig.add_trace(go.Scatter(x=[i] * neurons, y=[j for j in range(neurons)],
                                    mode='markers', marker=dict(size=30),
                                    text=[f'Neuron {j + 1}' for j in range(neurons)],
                                    name=f'Hidden Layer {i}'))

        fig.add_trace(go.Scatter(x=[n_layers + 1] * n_output, y=[k for k in range(n_output)],
                                mode='markers', marker=dict(size=30),
                                text=[f'Class {k + 1}' for k in range(n_output)], name='Output Layer'))

        fig.update_layout(title='Neural Network Architecture',
                          xaxis_title='Layers',
                          yaxis_title='Neurons',
                          showlegend=True)

        fig.show()