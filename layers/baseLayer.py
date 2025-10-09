import numpy as np


class baseLayer:
    def __init__(self, input_size, output_size, weight_initialize_type="he"):
        # todo initialize weights with some random protocol
        if weight_initialize_type == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                2.0 / input_size
            )
        elif weight_initialize_type == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                1.0 / input_size
            )
        else:
            self.weights = np.zeros((input_size, output_size))
            
        self.bias = np.zeros((1, output_size))

        # gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

        self.X_bar = None
        self.y_hat = None
