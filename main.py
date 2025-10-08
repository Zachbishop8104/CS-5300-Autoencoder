from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from gallary import plot_gallery
from costFunction import MSE
from layers.linearLayer import linearLayer
from layers.reluLayer import reluLayer
from layers.sigmoidLayer import outputLayer


def main():
    rng = RandomState(0)
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape
    
    # global centering
    faces_centered = faces - faces.mean(axis=0)
    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    batch_size = 40
    batches = []
    for i in range(0, n_samples, batch_size):
        batches.append(faces_centered[i : i + batch_size])

    learning_rate = 0.01
    n_epochs = 1

    # encoder-decoder architecture
    e1 = linearLayer(learning_rate, n_features, 300)
    e2 = reluLayer(learning_rate, 300, 100)
    e3 = linearLayer(learning_rate, 100, 50)
    e4 = reluLayer(learning_rate, 50, 10)

    d1 = linearLayer(learning_rate, 10, 50)
    d2 = reluLayer(learning_rate, 50, 100)
    d3 = linearLayer(learning_rate, 100, 300)
    d4 = outputLayer(learning_rate, 300, n_features)

    for batch in batches:
        for epoch in range(n_epochs):
            # forward pass
            out = e1.forward(batch)
            out = e2.forward(out)
            out = e3.forward(out)
            out = e4.forward(out)
            out = d1.forward(out)
            out = d2.forward(out)
            out = d3.forward(out)
            out = d4.forward(out)
            print("Output shape: ", out.shape)
            loss = MSE(batch, out)
            print("Loss: ", loss)

    print(np.array(batches).shape)

    n_row, n_col = 3, 4  # for plotting
    n_components = n_row * n_col  # for plotting
    image_shape = (64, 64)
    print("Dataset consists of %d faces" % n_samples)
    # plot_gallery("Faces from dataset", faces_centered[:n_components], n_col=n_col, n_row=n_row)


if __name__ == "__main__":
    main()
