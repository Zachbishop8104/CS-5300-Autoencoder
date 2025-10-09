from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from gallary import plot_gallery
from meanSquaredError import MSE
from layers.linearLayer import linearLayer
from layers.reluLayer import reluLayer
from layers.sigmoidLayer import sigmoidLayer
from layers.identityLayer import identityLayer


def main():
    n_row, n_col = 3, 4
    rng = RandomState(0)

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    # Train on the whole set
    train = faces

    # keeps in a nice range
    # could be removed possibly
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True) + 1e-6
    train_n = (train - mu) / sigma

    batch_size = 40
    learning_rate = 0.001
    n_epochs = 50

    # encoder-decoder
    e1 = linearLayer(n_features, 512)
    e2 = reluLayer()
    e3 = linearLayer(512, 128)
    e4 = reluLayer()

    d1 = linearLayer(128, 512)
    d2 = reluLayer()
    d3 = linearLayer(512, n_features, "xavier")
    d4 = sigmoidLayer()
    # d4 = identityLayer()

    mse = MSE()

    # small positive bias for linear layers
    for layer in (e1, e3, d1, d3):
        layer.bias += 0.01

    def forward_pass(X):
        z = e1.forward(X)
        z = e2.forward(z)
        z = e3.forward(z)
        z = e4.forward(z)
        z = d1.forward(z)
        z = d2.forward(z)
        z = d3.forward(z)
        z = d4.forward(z)
        return z

    last_batch = None
    last_recon = None

    for epoch in range(n_epochs):
        # shuffle each epoch
        perm = rng.permutation(len(train_n))
        train_shuf = train_n[perm]

        epoch_losses = []

        # iterate mini-batches
        for start in range(0, len(train_shuf), batch_size):
            batch = train_shuf[start : start + batch_size]

            # forward
            out = forward_pass(batch)

            # loss + grad
            loss = mse.forward(out, batch)
            grad = mse.backward()
            epoch_losses.append(loss)

            # backward
            grad = d4.backward(grad)
            grad = d3.backward(grad)
            grad = d2.backward(grad)
            grad = d1.backward(grad)
            grad = e4.backward(grad)
            grad = e3.backward(grad)
            grad = e2.backward(grad)
            _ = e1.backward(grad)

            # SGD update
            for layer in (e1, e3, d1, d3):
                layer.weights -= learning_rate * layer.dW
                layer.bias -= learning_rate * layer.db

            # if this is the final batch of the epoch, save it
            last_batch = batch
            last_recon = out

        print(
            f"Epoch {epoch+1}/{n_epochs} | mean train MSE: {np.mean(epoch_losses):.4f}"
        )

    # Show some of the freshest reconstructions
    k = n_row * n_col
    if last_recon.shape[0] > k:
        last_recon_to_show = last_recon[:k]
    else:
        last_recon_to_show = last_recon

    plot_gallery(
        "Reconstructed faces (freshest, final batch)",
        last_recon_to_show,
        n_col=n_col,
        n_row=n_row,
    )
    print(
        f"Displayed {last_recon_to_show.shape[0]} freshest reconstructions out of the final batch."
    )


if __name__ == "__main__":
    main()
