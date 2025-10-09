import numpy as np
from layers.baseLayer import baseLayer


class linearLayer(baseLayer):
    # in features is number of inputs
    # out features is number of outputs
    def __init__(self, in_features, out_features, weight_initialize_type="he"):
        super().__init__(in_features, out_features, weight_initialize_type)

    def forward(self, X):
        self.X_bar = X  # features going into layer
        self.y_hat = np.dot(X, self.weights) + self.bias
        return self.y_hat

    def backward(self, dY):
        # dY is from the previous layer
        # compute gradients
        self.dW = np.dot(self.X_bar.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)

        # compute gradient with respect to input features
        dX = np.dot(dY, self.weights.T)
        return dX
