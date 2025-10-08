class MSE:
    def __init__(self):
        self.diff = None
        self.scale = None

    def forward(self, pred, target):
        # pred, target: (B, D)
        self.diff = pred - target
        B, D = pred.shape
        self.scale = 1.0 / (B * D)
        return (self.diff ** 2).sum() * self.scale

    def backward(self):
        # dL/dpred
        return 2.0 * self.scale * self.diff
