import numpy as np
from layers.baseLayer import baseLayer


# for hidden layers
class reluLayer(baseLayer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, X):
        self.X_bar = X
        self.y_bar = np.maximum(0, X @ self.weights + self.bias)
        return self.y_bar

    def backward(self, dY):
        pass
