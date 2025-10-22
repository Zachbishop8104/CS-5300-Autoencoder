import numpy as np


class linearLayer():
    # in features is number of inputs
    # out features is number of outputs
    def __init__(self, in_features, out_features, weight_initialize_type="he"):
        if weight_initialize_type == "he":
            self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif weight_initialize_type == "xavier":
            self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        else:
            self.weights = np.random.randn(in_features, out_features) * 0.01

        self.bias = np.zeros((1, out_features))

        # gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

        self.X_bar = None
        self.y_hat = None

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
