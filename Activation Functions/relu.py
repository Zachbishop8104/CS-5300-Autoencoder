from base import Module
import numpy as np

#for hidden layers
class ReLU(Module):
    def __init__(self): 
        self.mask = None
        
    def forward(self, X):
        self.mask = (X > 0)
        return np.where(self.mask, X, 0.0)
    
    def backward(self, dY):
        return dY * self.mask.astype(dY.dtype)