import numpy as np

class linearLayer:
    #in features is number of inputs
    #out features is number of outputs
    def __init__(self, in_features, out_features):
        #todo initialize weights with some random protocol
        self.weights = np.zeros((in_features, out_features))
        self.bias = np.zeros((1, out_features))
        
        #gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        
        self.X_bar = None

    def forward(self, batch):
        self.X_bar = batch #features going into layer
        return np.dot(batch, self.weights) + self.bias
    
    def backward(self, dY):
        #dY is the gradient of the loss with respect to the output of this layer
        #compute gradients
        self.dW = np.dot(self.X_bar.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)

        #compute gradient with respect to input features
        dX = np.dot(dY, self.weights.T)
        return dX
    