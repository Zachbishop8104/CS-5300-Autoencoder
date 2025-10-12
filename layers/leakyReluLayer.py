import numpy as np

class leakyReluLayer:
    def __init__(self, alpha=0.001):
        self.alpha = alpha

    def forward(self, X):
        self.X_bar = X
        self.y_hat = np.where(X > 0, X, X * self.alpha)
        
        return self.y_hat

    def backward(self, dY):
        dX = dY * np.where(self.X_bar > 0, 1.0, self.alpha)
        return dX