import math


class Neuron():
    def __init__(self):
        self.x = []
        self.w = []
        self.sum = 0
        self.y = 0

    def add_weights(self, *args):
        self.w.extend(args)

    def add_x(self, *args):
        self.x.extend(args)

    def summator(self):
        for i in range(len(self.x)):
            self.sum += self.x[i] * self.w[i]

    def activation_func(self, z):
        if self.sum >= z:
            self.y = 1
        else:
            self.y = 0
        print(self.y)

    def sigmoid_func(self):
        func = 1 / (1 + math.exp(-self.sum))
        if func > 0.99:
            self.y = 1
        elif func < 0.01:
            self.y = 0
        else:
            self.y = func
        print(self.y)


network = Neuron()
network.add_x(1, 1)
network.add_weights(-2, 5)

network.summator()

network.activation_func(10)
network.sigmoid_func()
