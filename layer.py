import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)
        self.biases = 0.1 * np.random.randn(1, n_outputs)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
