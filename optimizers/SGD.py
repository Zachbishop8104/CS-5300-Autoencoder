
class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.weights -= self.lr * layer.dW
            layer.bias -= self.lr * layer.db