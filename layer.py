import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros(n_outputs)
        self.learning_rate = 0.1

    def forward(self, inputs):
        self.old_x = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)
        self.grad = np.dot(grad, self.weights.transpose())

    def update(self):
        self.weights = self.weights - self.learning_rate * self.grad_w
        self.biases = self.weights - self.learning_rate * self.grad_b

    def transpose(self):
        self.weights = self.weights.transpose()
        self.biases = self.biases.transpose()
