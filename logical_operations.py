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

    def and_func(self, x1=1, x2=1):
        self.neuron.clear_all()

        w1 = 2
        w2 = 2
        b = -3
        self.neuron.add_x(x1, x2)
        self.neuron.add_weights(w1, w2)

        self.neuron.summator(b)

        self.neuron.step_func()


operation = LogicalOperations()

operation.not_func(1)
operation.and_func(1, 1)
