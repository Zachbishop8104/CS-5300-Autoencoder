import numpy as np

class sigmoidLayer():
    def __init__(self):
        pass

    def forward(self, X):
        self.X_bar = X
        self.y_hat = 1 / (1 + np.exp(-X))
        return self.y_hat

    def backward(self, dY):
        return dY * (self.y_hat * (1 - self.y_hat))
