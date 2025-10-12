import numpy as np

class sigmoidLayer():
    def __init__(self):
        pass

    def forward(self, X):
        self.X_bar = X
        Z = np.clip(X, -50, 50)  # stability / flooding warning --might not be needed
        self.y_hat = 1.0 / (1.0 + np.exp(-Z))
        return self.y_hat

    def backward(self, dY):
        return dY * (self.y_hat * (1 - self.y_hat))
