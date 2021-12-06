import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))


class Activation_Step:
    # сначала округляем в большую сторону и спасаем все значения X > 0
    # потом все значения приравниваем к единице
    def forward(self, inputs):
        self.output = np.minimum(1, np.maximum(0, np.ceil(inputs)))
