import numpy as np

#for hidden layers
class reluLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.zeros((in_features, out_features))
        self.bias = np.zeros((1, out_features))
        
        #gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        
        self.X_bar = None