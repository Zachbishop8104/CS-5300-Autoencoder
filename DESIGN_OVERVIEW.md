## Design Overview

This document provides a high-level design for the CS-5300 Autoencoder project, including detailed pseudocode, component descriptions, packages/tools, and interfaces. It covers layers, optimizers, dataset handling, and the loss function to assemble and extend the project.

### Architecture Summary
- Data: Olivetti Faces (64×64 grayscale), loaded via scikit-learn and grouped by subject for leave-one-subject-out cross-validation (40 folds).
- Model: Fully-connected autoencoder with an MLP encoder and mirrored decoder.
- Loss: Mean Squared Error (MSE) between input and reconstruction.
- Optimizers: Adam or SGD operating over trainable `linearLayer` parameters.
- Visualization: Grid plots of original and reconstructed faces.

### Pseudocode (End-to-End)

```pseudo
function train_and_evaluate(encoder_sizes, epochs, batch_size, optimizer_config, activation_config, seed, plot, verbose):
    rng ← RandomState(seed)
    faces, targets ← fetch_olivetti_faces(..., random_state=rng)
    n_samples, n_features ← shape(faces)

    subjects ← unique(targets)
    groups ← [faces[targets == s] for s in subjects]  # 40 groups × 10 images each

    mse ← new MSE()
    val_mses ← []

    for fold in range(len(groups)):
        validation_set ← groups[fold]
        training_set ← vstack(groups[:fold] + groups[fold+1:])

        layers, optimizer, bottleneck_dim ← initialize_model(
            num_features=n_features,
            encoder_sizes=encoder_sizes,
            optimizer_config=optimizer_config,
            activation_config=activation_config
        )

        for epoch in range(epochs):
            epoch_losses ← []
            train_shuf ← shuffle(training_set, rng)

            for batch in iterate_minibatches(train_shuf, batch_size):
                out ← forward_pass(layers, batch)
                loss ← mse.forward(out, batch)
                grad ← mse.backward()
                epoch_losses.append(loss)

                backward_pass(layers, grad)
                optimizer.step()

            if verbose:
                print(mean(epoch_losses))

        val_out ← forward_pass(layers, validation_set)
        val_loss ← mse.forward(val_out, validation_set)
        val_mses.append(val_loss)

        if plot:
            show_gallery(title="Fold reconstructions", images=stack(validation_set[:5], val_out[:5]))

    return {
        mean_train_mse_last_epoch: mean(epoch_losses),
        mean_val_mse: mean(val_mses),
        compression_ratio: n_features / bottleneck_dim,
        bottleneck_dim: bottleneck_dim,
        n_features: n_features
    }


function initialize_model(num_features, encoder_sizes, optimizer_config, activation_config):
    act ← build_activation(activation_config)  # relu or leaky_relu
    layers ← []

    # encoder
    in_dim ← num_features
    for size in encoder_sizes:
        layers.append(Linear(in_dim, size, init="he"))
        layers.append(act())
        in_dim ← size

    bottleneck_dim ← in_dim

    # decoder (mirror, excluding first encoder layer size)
    for size in reverse(encoder_sizes[:-1]):
        layers.append(Linear(in_dim, size, init="he"))
        layers.append(act())
        in_dim ← size

    layers.append(Linear(in_dim, num_features, init="xavier"))
    layers.append(Sigmoid())

    for L in layers:
        if has_attr(L, bias):
            L.bias ← L.bias + 0.01

    trainable_layers ← [L for L in layers if has_attr(L, weights)]
    optimizer ← build_optimizer(optimizer_config, trainable_layers)

    return layers, optimizer, bottleneck_dim


function forward_pass(layers, X):
    z ← X
    for layer in layers:
        z ← layer.forward(z)
    return z


function backward_pass(layers, grad):
    dY ← grad
    for layer in reverse(layers):
        dY ← layer.backward(dY)


function build_optimizer(cfg, trainable_layers):
    if cfg.type == "adam":
        return Adam(layers=trainable_layers, lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps, weight_decay=cfg.weight_decay, adamw=cfg.adamw)
    else if cfg.type == "sgd":
        return SGD(layers=trainable_layers, lr=cfg.lr)
    else:
        raise ValueError("unsupported optimizer type")
```

### Components and Interfaces

#### Dataset (scikit-learn)
- Source: `sklearn.datasets.fetch_olivetti_faces` with `data_home="data/olivetti_faces"`.
- Shape: `(n_samples=400, n_features=4096)` where 4096 = 64×64.
- Splitting: Leave-one-subject-out; 40 folds with 10 images per subject.
- Interface points:
  - Used in `main.train_and_evaluate` for data retrieval, shuffling, and folding.

#### Loss Function: `meanSquaredError.MSE`
- Methods:
  - `forward(pred, target) -> float`: Computes MSE averaged across batch and feature dimension.
  - `backward() -> ndarray`: Returns gradient with respect to `pred` used to seed backprop.
- State:
  - Caches `diff` and per-batch `scale` for gradient.

#### Layers (NumPy-based)
- `linearLayer(in_features, out_features, weight_initialize_type)`
  - Attributes: `weights`, `bias`, gradients `dW`, `db`.
  - `forward(X) -> Y = X·W + b`
  - `backward(dY) -> dX`, accumulates `dW = X^T·dY`, `db = sum(dY)`
  - Init: He by default, Xavier optional (used for final layer).

- `reluLayer`
  - `forward(X) -> max(0, X)`
  - `backward(dY) -> dY * (X > 0)`

- `leakyReluLayer(alpha=0.001)`
  - `forward(X) -> X if X>0 else alpha*X`
  - `backward(dY) -> dY * (1 if X>0 else alpha)`

- `sigmoidLayer`
  - `forward(X) -> 1 / (1 + exp(-clip(X)))`
  - `backward(dY) -> dY * sigmoid(X) * (1 - sigmoid(X))` using cached output

- Composition:
  - Model is a list of layers; `forward_pass` and `backward_pass` iterate in order and reverse order respectively.

#### Optimizers
- `optimizers.adamOptimizer.Adam`
  - Init: `(layers, lr, beta1, beta2, eps, weight_decay, adamw)`
  - Maintains first/second moments for each layer's `weights` and `bias`.
  - `step()`: Bias-corrected updates; optional decoupled weight decay when `adamw=True`.

- `optimizers.sgd.SGD`
  - Init: `(layers, lr)`
  - `step()`: Vanilla SGD update on `weights` and `bias`.

#### Visualization
- `gallary.plot_gallery(title, images, n_col, n_row, cmap)`
  - Expects flattened vectors; reshapes to 64×64 and displays grids.

#### Experiment Runner
- `run_test.py`
  - Reads a JSON config (single object or list), builds `optimizer_config` and `activation_config`, and calls `train_and_evaluate`.
  - Outputs machine-readable JSON with metrics and echo of the configuration.

### Packages and Tools
- NumPy: tensor ops and parameters.
- scikit-learn: Olivetti dataset loader.
- Matplotlib: plotting grids of reconstructions.

### Extension Points
- Change architecture by editing `encoder_sizes` and activation type (`relu` or `leaky_relu`).
- Swap loss (e.g., MAE) by implementing a new loss with `forward/backward`.
- Add optimizers by exposing a `step()` API operating on layers with `weights/bias/dW/db`.
- Add regularization or normalization layers following the same layer interface.

### Invariants and Assumptions
- Inputs are normalized to [0, 1]; output uses `sigmoid` to match this range.
- Gradients are computed per-batch; optimizers consume `dW`, `db` from `linearLayer` instances.
- Final layer uses Xavier init to stabilize `sigmoid` output.


