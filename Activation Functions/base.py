class Module:
    def params(self): return []
    def grads(self):  return []
    def zero_grad(self):
        for g in self.grads(): g.fill(0.0)