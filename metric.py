import numpy as np


class Metric_Accuracy:
    def calculate(self, output, y):
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy
