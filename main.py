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
from optimizers.adamOptimizer import Adam
from optimizers.SGD import SGD

def initialize_model(num_features, use_adam=True, learning_rate=0.001):
    # Initialize layers
    layers = []
    # encoder
    layers.append(linearLayer(num_features, 512))
    layers.append(leakyReluLayer())
    layers.append(linearLayer(512, 128))
    layers.append(leakyReluLayer())

    # decoder
    layers.append(linearLayer(128, 512))
    layers.append(leakyReluLayer())
    layers.append(linearLayer(512, num_features, "xavier"))
    layers.append(sigmoidLayer())

    # small positive bias for linear layers
    for layer in (layers[0], layers[2], layers[4], layers[6]):
        layer.bias += 0.01
        
    # optimizers
    opt = None
    if use_adam:
        opt = AdamOpt = Adam(
            layers=(layers[0], layers[2], layers[4], layers[6]),
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=1e-5,
            adamw=True,
        )
    else:
        opt = SGDOpt = SGD(
            layers=(layers[0], layers[2], layers[4], layers[6]),
            lr=learning_rate,
        )
    
    return layers, opt


def forward_pass(layers, X):
    z = X
    for layer in layers:
        z = layer.forward(z)
        
    return z

def backward_pass(layers, grad):
    dY = grad
    for layer in reversed(layers):
        dY = layer.backward(dY)


def main():
    # grab faces from selected seed
    rng = RandomState(0)
    faces, targets = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng) # 400 faces
    n_samples, n_features = faces.shape
    subjects = np.unique(targets)
    groups = [faces[targets == s] for s in subjects]  # each is shape (10, 4096)
    
    epochs = 50
    batch_size = 35
    
    folds = len(groups)  # 40
    for fold in range(folds):
        validation_set = groups[fold]
        training_set = np.vstack(groups[:fold] + groups[fold+1:]) # combine all other groups for training (390 faces. 390, 4096)
        
        # create model
        layers, opt = initialize_model(num_features=n_features, use_adam=True, learning_rate=0.001)
        mse = MSE()
        
        for epoch in range(epochs):
            epoch_losses = []
            perm = rng.permutation(len(training_set))
            train_shuf = training_set[perm]
            
            for start in range(0, len(train_shuf), batch_size):
                batch = train_shuf[start : start + batch_size]
                
                # forward
                out = forward_pass(layers, batch)
                
                # loss + grad
                loss = mse.forward(out, batch)
                grad = mse.backward()
                epoch_losses.append(loss)
                
                # backward
                backward_pass(layers, grad)
                opt.step()
                
            print(f"Epoch {epoch+1}/{epochs} | mean train MSE: {np.mean(epoch_losses):.4f}")
            
        print(f"\nCompleted fold {fold+1}/{folds}.\n")
        
        # now that we are done with this fold lets test on the validation set
        val_out = forward_pass(layers, validation_set)
        val_loss = mse.forward(val_out, validation_set)
        print(f"Validation MSE for fold {fold+1}/{folds}: {val_loss:.4f}\n")
        
        # output the original and reconstructed faces for this fold
        show_num = 5
        imgs = np.vstack((validation_set[:show_num], val_out[:show_num]))  # 10 images = 2 rows × 5 cols
        plot_gallery(
            f"Fold {fold+1} Reconstructions",
            imgs,
            n_row=2,
            n_col=5,
        )
        

if __name__ == "__main__":
    main()
