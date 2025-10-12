from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from gallary import plot_gallery
from meanSquaredError import MSE
from layers.linearLayer import linearLayer
from layers.reluLayer import reluLayer
from layers.sigmoidLayer import sigmoidLayer
from layers.identityLayer import identityLayer
from layers.leakyReluLayer import leakyReluLayer
from adamOptimizer import Adam


def main():
    n_row, n_col = 3, 3
    rng = RandomState(0)

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    # Train on the whole set
    train = faces

    # I was getting some pretty close looking faces with setting the data like this but I don't think it is correct, -- just learned that this is more for a identity layer output?
    # keeps in a nice range
    # could be removed possibly
    # mu = train.mean(axis=0, keepdims=True)
    # sigma = train.std(axis=0, keepdims=True) + 1e-6
    # train_n = (train - mu) / sigma

    batch_size = 35
    learning_rate = 0.001
    n_epochs = 150

    # encoder-decoder
    e1 = linearLayer(n_features, 512)
    e2 = leakyReluLayer()
    e3 = linearLayer(512, 128)
    e4 = leakyReluLayer()

    d1 = linearLayer(128, 512)
    d2 = leakyReluLayer()
    d3 = linearLayer(512, n_features, "xavier")
    d4 = sigmoidLayer()
    # d4 = identityLayer()

    # new optimizer instead of SGD, made everything way better but slower to train
    opt = Adam(
        layers=(e1, e3, d1, d3),
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-5,
        adamw=True,
    )

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
        perm = rng.permutation(len(train))
        train_shuf = train[perm]

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
            # for layer in (e1, e3, d1, d3):
            #     layer.weights -= learning_rate * layer.dW
            #     layer.bias -= learning_rate * layer.db

            opt.step()

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

    print(
        f"Displayed {last_recon_to_show.shape[0]} freshest reconstructions out of the final batch."
    )

    plot_gallery(
        "Reconstructed faces (freshest, final batch)",
        last_recon_to_show,
        n_col=n_col,
        n_row=n_row,
    )


if __name__ == "__main__":
    main()
