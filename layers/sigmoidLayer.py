import math
from layers.baseLayer import baseLayer


class outputLayer(baseLayer):
    def __init__(self, learning_rate, in_features, out_features):
        super().__init__(learning_rate, in_features, out_features)

    def forward(self, X):
        return 1/(1 + math.exp(X @ self.weights + self.bias))

    def backward(self, dY):
        pass
