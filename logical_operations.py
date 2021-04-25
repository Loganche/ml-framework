import basic_neuron as bn


class LogicalOperations():
    def __init__(self):
        self.neuron = bn.Neuron()

    def not_func(self, x=1):
        w = -2
        b = 1
        self.neuron.add_x(x)
        self.neuron.add_weights(w)

        self.neuron.summator(b)

        self.neuron.step_func()


def test():
    not_operation = LogicalOperations()
    not_operation.not_func(1)


# print(test())
