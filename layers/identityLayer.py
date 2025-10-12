class identityLayer:
    def __init__(self):
        pass
    
    def forward(self, X):
        self.X = X
        return X
    
    def backward(self, dY):
        return dY
