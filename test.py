import numpy as np
import matplotlib.pyplot as plt


import layer
import activation
import loss
import metric

'''
 nn_basics
    --basic_neuron.py
    --logical_operations.py

import nn_basics.basic_neuron as bn
import nn_basics.logical_operations as lo

network = bn.Neuron()
network.add_x(1, 1)
network.add_weights(-2, 5)

network.summator()

network.step_func()
print(f'Step function: {network.y}')
network.sigmoid_func()
print(f'Sigmoid function: {network.y}')


operation = lo.LogicalOperations()

print(f'NOT: {operation.not_func(1)}')
print(f'AND: {operation.and_func(1, 1)}')
print(f'OR: {operation.or_func(0, 0)}')
print(f'XOR: {operation.xor_func(0, 1)}')
'''

'''  layer.py activation.py '''

# INPUT
# https://cs231n.github.io/neural-networks-case-study/
N = 20  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# Model Architecture
layer1 = layer.Layer_Dense(2, 4)
activation1 = activation.Activation_LeakyReLU()
layer2 = layer.Layer_Dense(4, 3)
activation2 = activation.Activation_Softmax()
loss1 = loss.Loss_CategoricalCrossEntropy()
metric1 = metric.Metric_Accuracy()

# Model Evaluation
layer1.forward(X)  # FC layer
activation1.forward(layer1.output)  # Non-linear fun
layer2.forward(activation1.output)  # FC layer
activation2.forward(layer2.output)  # Non-linear fun
loss1.calculate(activation2.output, y)  # Loss fun
print(f"Loss: {loss1.calculate(activation2.output, y)}")  # Results
print(f"Accuracy: {metric1.calculate(activation2.output, y)}")  # Metrics
