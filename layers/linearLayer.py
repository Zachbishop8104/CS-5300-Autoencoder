import numpy as np
from layers.baseLayer import baseLayer


class linearLayer(baseLayer):
    # in features is number of inputs
    # out features is number of outputs
    def __init__(self, learning_rate, in_features, out_features):
        super().__init__(learning_rate, in_features, out_features)

    def forward(self, batch):
        self.X_bar = batch  # features going into layer
        return np.dot(batch, self.weights) + self.bias

    def backward(self, dY):
        # dY is the gradient of the loss with respect to the output of this layer
        # compute gradients
        self.dW = np.dot(self.X_bar.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)

        # compute gradient with respect to input features
        dX = np.dot(dY, self.weights.T)
        return dX
