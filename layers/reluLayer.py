import numpy as np

# for hidden layers
class reluLayer():
    def __init__(self):
        pass

    def forward(self, X):
        self.X_bar = X
        self.y_bar = np.maximum(0, X)
        return self.y_bar

    def backward(self, dY):
        dZ = dY * (self.X_bar > 0).astype(float)
        return dZ
        