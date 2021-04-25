import basic_neuron as bn


class LogicalOperations():
    def __init__(self):
        self.neuron = bn.Neuron()

    def not_func(self, x=1):
        self.neuron.clear_all()

        w = -2
        b = 1
        self.neuron.add_x(x)
        self.neuron.add_weights(w)

        self.neuron.summator(b)
        self.neuron.step_func()

        return self.neuron.y

    def and_func(self, x1=1, x2=1):
        self.neuron.clear_all()

        w1 = 2
        w2 = 2
        b = -3
        self.neuron.add_x(x1, x2)
        self.neuron.add_weights(w1, w2)

        self.neuron.summator(b)
        self.neuron.step_func()

        return self.neuron.y

    def or_func(self, x1=0, x2=0):
        self.neuron.clear_all()

        w1 = 2
        w2 = 2
        b = -1
        self.neuron.add_x(x1, x2)
        self.neuron.add_weights(w1, w2)

        self.neuron.summator(b)
        self.neuron.step_func()

        return self.neuron.y

    def xor_func(self, x1=1, x2=1):
        # X1 XOR X2 = (NOT X1 AND X2) OR (X1 AND NOT X2)
        self.neuron.clear_all()

        NOT_X1 = self.not_func(x1)
        NOT_X2 = self.not_func(x2)

        AND_NOT_X1 = self.and_func(NOT_X1, x2)
        AND_NOT_X2 = self.and_func(x1, NOT_X2)

        self.or_func(AND_NOT_X1, AND_NOT_X2)

        return self.neuron.y


''' Test
operation = LogicalOperations()

print(f'NOT: {operation.not_func(1)}')
print(f'AND: {operation.and_func(1, 1)}')
print(f'OR: {operation.or_func(0, 0)}')
print(f'XOR: {operation.xor_func(0, 1)}')
'''
