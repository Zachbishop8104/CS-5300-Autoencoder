from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from gallary import plot_gallery
from meanSquaredError import MSE
from layers.linearLayer import linearLayer
from layers.reluLayer import reluLayer
from layers.sigmoidLayer import sigmoidLayer


def main():
    n_row, n_col = 3, 4
    n_components = n_row * n_col
    image_shape = (64, 64)

    rng = RandomState(0)
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    batch_size = 40
    batches = [faces[i : i + batch_size] for i in range(0, n_samples, batch_size)]

    learning_rate = 0.05
    n_epochs = 10

    # encoder-decoder
    e1 = linearLayer(n_features, 200)
    e2 = reluLayer()
    e3 = linearLayer(200, 50)
    e4 = reluLayer()

    d1 = linearLayer(50, 200)
    d2 = reluLayer()
    d3 = linearLayer(200, n_features)
    d4 = sigmoidLayer() 

    mse = MSE()

    for i, batch in enumerate(batches):
        for epoch in range(n_epochs):
            # forward
            out = e1.forward(batch)
            out = e2.forward(out)
            out = e3.forward(out)
            out = e4.forward(out)
            out = d1.forward(out)
            out = d2.forward(out)
            out = d3.forward(out)
            out = d4.forward(out)

            loss = mse.forward(out, batch)
            grad = mse.backward()

            # backward
            grad = d4.backward(grad)
            grad = d3.backward(grad)
            grad = d2.backward(grad)
            grad = d1.backward(grad)
            grad = e4.backward(grad)
            grad = e3.backward(grad)
            grad = e2.backward(grad)
            _ = e1.backward(grad)

            # update params
            for layer in [e1, e3, d1, d3]:
                layer.weights -= learning_rate * layer.dW
                layer.bias -= learning_rate * layer.db

        print(f"Batch {i}, epoch {epoch}, loss {loss:.4f}")
        plot_gallery(
            f"Reconstructed faces (batch {i}, epoch {epoch})",
            out[:n_components],
            n_col=n_col,
            n_row=n_row,
        )

    print("Dataset consists of %d faces" % n_samples)


if __name__ == "__main__":
    main()
