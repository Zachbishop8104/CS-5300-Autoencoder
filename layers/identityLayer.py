class identityLayer:
    def forward(self, X):
        self.X = X
        return X
    def backward(self, dY):
        return dY
