import numpy as np


# Just dropped this in, don't really understand right now but will figure it out later since this made the output almost perfect
class Adam:
    def __init__(
        self,
        layers,
        lr=5e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        adamw=True
    ):
        """
        layers: iterable of layers with .weights, .bias, .dW, .db
        adamw:  True = decoupled weight decay (AdamW), False = L2 inside grads
        """
        self.layers = list(layers)
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.adamw = adamw
        self.t = 0

        # Per-parameter moment buffers
        self.mW = {id(L): np.zeros_like(L.weights) for L in self.layers}
        self.vW = {id(L): np.zeros_like(L.weights) for L in self.layers}
        self.mb = {id(L): np.zeros_like(L.bias) for L in self.layers}
        self.vb = {id(L): np.zeros_like(L.bias) for L in self.layers}

    def step(self):
        self.t += 1
        b1t = self.b1**self.t
        b2t = self.b2**self.t

        for L in self.layers:
            key = id(L)

            # Optionally add L2 to grads (classic Adam-L2). Usually prefer AdamW.
            dW = L.dW + (0.0 if self.adamw else self.wd) * L.weights
            db = L.db  # typically do NOT decay biases

            # Update first/second moments
            self.mW[key] = self.b1 * self.mW[key] + (1 - self.b1) * dW
            self.vW[key] = self.b2 * self.vW[key] + (1 - self.b2) * (dW * dW)
            self.mb[key] = self.b1 * self.mb[key] + (1 - self.b1) * db
            self.vb[key] = self.b2 * self.vb[key] + (1 - self.b2) * (db * db)

            # Bias correction
            mW_hat = self.mW[key] / (1 - b1t)
            vW_hat = self.vW[key] / (1 - b2t)
            mb_hat = self.mb[key] / (1 - b1t)
            vb_hat = self.vb[key] / (1 - b2t)

            # Parameter updates
            L.weights -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps))
            L.bias -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))

            # Decoupled weight decay (AdamW)
            if self.adamw and self.wd > 0.0:
                L.weights -= self.lr * self.wd * L.weights
                # (Usually skip decay on biases)
