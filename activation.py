import numpy as np


class Activation_Softmax:
    '''
    A generalized version [0;1] of Sigmoid activation for Logistic Multiclass Classifier.
    Mostly used in the Output Layer.
    '''

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, grad):
        self.grad = self.output * (grad - (grad * self.output).sum(axis=1, keepdims=True))


class Activation_Sigmoid:
    '''
    An activation function [0;1] for Logistic Binary Classifier.
    Mostly used in the Output Layer.
    '''

    def forward(self, inputs):
        self.output = np.exp(inputs) / (1 + np.exp(inputs))

    def backward(self, grad):
        self.grad = self.output * (1. - self.output) * grad


class Activation_ReLU:
    '''
    An activation function [0;inf] simplier then Tanh or Sigmoid.
    Mostly used in Any Layers.
    '''

    def forward(self, inputs):
        self.old_x = np.copy(inputs)
        self.output = np.maximum(0, inputs)

    def backward(self, grad):
        self.grad = np.where(self.old_x > 0, grad, 0)


class Activation_LeakyReLU:
    '''
    An improved version of ReLU [;inf] saving negative 0.01*X values.
    Mostly used in Any Layers.
    '''

    def forward(self, inputs):
        self.old_x = np.copy(inputs)
        self.output = np.where(inputs > 0, inputs, 0.01 * inputs)

    def backward(self, grad):
        self.grad = np.where(self.old_x > 0, grad, 0.01 * grad)


class Activation_Tanh:
    '''
    An activation function [-1;1].
    Mostly used in the Hidden Layer.
    '''

    def forward(self, inputs):
        self.output = np.tanh(inputs)


class Activation_Step:
    '''
    An activation function [-1;1] used in early years of ML.
    Mostly used in Any Layers.
    '''

    def forward(self, inputs):
        self.output = np.minimum(1, np.maximum(0, np.ceil(inputs)))

    def backward(self, grad):
        self.grad = np.where(self.old_x != 0, 0, None)
