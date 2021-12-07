import numpy as np


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        self.old_y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        self.old_y_true = y_true
        if len(y_true.shape) == 1:
            conf = self.old_y_pred[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            conf = np.sum(self.old_y_pred * y_true, axis=1)
        self.output = -np.log(conf)
        return self.output

    def backward(self):
        self.grad = np.where(self.old_y_true == 1, -1 / self.old_y_pred, 0)
