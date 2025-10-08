import numpy as np


class baseLayer:
    def __init__(self, learning_rate, input_size, output_size):
        # todo initialize weights with some random protocol
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.learning_rate = learning_rate

        # gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

        self.X_bar = None
