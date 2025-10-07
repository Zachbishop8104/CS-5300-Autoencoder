import numpy as np
from base import Module

#for output layer (0-1) probably
class Sigmoid(Module):
    def __init__(self):
        self.out = None
    
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out
    
    def backward(self, dY):
        return dY * self.out * (1 - self.out)