import numpy as np

class MSE:
    def __init__(self):
        self.diff = None
        self.scale = None

    def forward(self, pred, target):
        # pred, target: (B, D)
        self.diff = pred - target
        B, D = pred.shape
        self.scale = 1.0 / B
        return np.sum(self.diff * self.diff) * self.scale / D

    def backward(self):
        # dL/dpred
        return 2.0 * self.scale * self.diff
