import numpy as np
import matplotlib.pyplot as plt


import layer
import activation
import loss
import metric
import data

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
X, y = data.data_spiral()
y_onehot = data.one_hot(y)

'''
print(f"X: {X}")
print(f"y: {y}")
print(f"y_onehot: {y_onehot}")
'''

# Model Architecture
layer1 = layer.Layer_Dense(2, 4)
activation1 = activation.Activation_LeakyReLU()
layer2 = layer.Layer_Dense(4, 3)
activation2 = activation.Activation_Softmax()
loss1 = loss.Loss_CategoricalCrossEntropy()
metric1 = metric.Metric_Accuracy()

# Model Evaluation
layer1.forward(X)  # FC layer
weights1 = layer1.weights
activation1.forward(layer1.output)  # Non-linear fun
layer2.forward(activation1.output)  # FC layer
activation2.forward(layer2.output)  # Non-linear fun
loss1.forward(activation2.output, y_onehot)  # Loss fun

# print(f"Output: {activation2.output}")  # Results
print(f"Loss: {np.mean(loss1.output)}")  # Loss
print(f"Accuracy: {metric1.calculate(activation2.output, y)}")  # Metrics


# Model Backpropogation
loss1.backward()

activation2.backward(loss1.grad)
layer2.backward(activation2.grad)
layer2.update()

activation1.backward(layer2.grad)
layer1.backward(activation1.grad)
layer1.update()


# Model Evaluation
layer1.forward(X)  # FC layer
activation1.forward(layer1.output)  # Non-linear fun
layer2.forward(activation1.output)  # FC layer
activation2.forward(layer2.output)  # Non-linear fun
loss1.forward(activation2.output, y_onehot)  # Loss fun

# print(f"Output: {activation2.output}")  # Results
print(f"Loss: {np.mean(loss1.output)}")  # Loss
print(f"Accuracy: {metric1.calculate(activation2.output, y)}")  # Metrics
