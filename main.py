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

def initialize_model(
    num_features,
    encoder_sizes=None,
    optimizer_config=None,
    activation_config=None,
):
    # defaults
    if encoder_sizes is None or len(encoder_sizes) == 0:
        encoder_sizes = [512, 128]
    if optimizer_config is None:
        optimizer_config = {"type": "adam", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-5, "adamw": True}
    if activation_config is None:
        activation_config = {"type": "relu"}

    # Choose activation
    act_type = str(activation_config.get("type", "relu")).lower()
    alpha = float(activation_config.get("alpha", 0.001))
    def activation_factory():
        if act_type == "leaky_relu":
            return leakyReluLayer(alpha=alpha)
        return reluLayer()

    # Initialize layers
    layers = []

    # encoder
    in_dim = num_features
    for size in encoder_sizes:
        layers.append(linearLayer(in_dim, size))
        layers.append(activation_factory())
        in_dim = size

    bottleneck_dim = in_dim

    # decoder mirrors the encoder (excluding the first encoder layer size)
    for size in reversed(encoder_sizes[:-1]):
        layers.append(linearLayer(in_dim, size))
        layers.append(activation_factory())
        in_dim = size

    # final output layer
    layers.append(linearLayer(in_dim, num_features, "xavier"))
    layers.append(sigmoidLayer())

    # small positive bias for all linear layers
    for L in layers:
        if hasattr(L, "bias"):
            L.bias += 0.01

    # collect trainable (linear) layers
    trainable_layers = tuple([L for L in layers])

    # optimizer setup
    opt = None
    opt_type = str(optimizer_config.get("type", "adam")).lower()
    if opt_type == "adam":
        opt = Adam(
            layers=trainable_layers,
            lr=optimizer_config.get("lr", 1e-3),
            beta1=optimizer_config.get("beta1", 0.9),
            beta2=optimizer_config.get("beta2", 0.999),
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 1e-5),
            adamw=optimizer_config.get("adamw", True),
        )
    else:
        opt = SGD(
            layers=trainable_layers,
            lr=optimizer_config.get("lr", 1e-2),
        )

    return layers, opt, bottleneck_dim


def forward_pass(layers, X):
    z = X
    for layer in layers:
        z = layer.forward(z)
        
    return z

def backward_pass(layers, grad):
    dY = grad
    for layer in reversed(layers):
        dY = layer.backward(dY)

def train_and_evaluate(
    encoder_sizes=None,
    epochs=50,
    batch_size=35,
    optimizer_config=None,
    activation_config=None,
    seed=0,
    plot=True,
    verbose=True,
    cv_type="loso",
    n_splits=5,
):
    # grab faces from selected seed
    rng = RandomState(seed)
    faces, targets = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng, data_home="data/olivetti_faces")
    n_samples, n_features = faces.shape
    subjects = np.unique(targets)
    groups = [faces[targets == s] for s in subjects]  # each is shape (10, 4096)

    mse = MSE()
    val_mses = []
    last_epoch_mean_train_mse = None
    bottleneck_dim_out = None

    last_validation = None
    last_val_out = None

    # Build folds: LOSO (leave-one-subject-out) or K-Fold by subject
    cv_mode = str(cv_type).lower()
    indices = np.arange(len(groups))
    if cv_mode == "kfold":
        m = len(indices)
        k = int(n_splits) if int(n_splits) > 1 else 2
        if k > m:
            k = m
        # shuffle subject indices deterministically
        perm = RandomState(seed).permutation(m)
        # compute fold sizes (almost even split)
        base = m // k
        rem = m % k
        fold_sizes = [base + (1 if i < rem else 0) for i in range(k)]
        # slice permutation into folds
        folds = []
        start = 0
        for size in fold_sizes:
            stop = start + size
            folds.append(perm[start:stop])
            start = stop
        total_folds = k
        def _kfold_iter():
            for i in range(k):
                val_idx = np.array(folds[i])
                train_idx = np.array(np.concatenate([folds[j] for j in range(k) if j != i]))
                yield (train_idx, val_idx)
        fold_iter = _kfold_iter()
    else:
        # Default: LOSO over subjects
        total_folds = len(groups)
        def _loso_iter():
            for i in range(len(groups)):
                val_idx = np.array([i])
                train_idx = np.array([j for j in range(len(groups)) if j != i])
                return_idx = (train_idx, val_idx)
                yield return_idx
        fold_iter = _loso_iter()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
        # stack subjects into train/val arrays
        training_set = np.vstack([groups[i] for i in train_idx])
        validation_set = np.vstack([groups[i] for i in val_idx])

        layers, opt, bottleneck_dim = initialize_model(
            num_features=n_features,
            encoder_sizes=encoder_sizes,
            optimizer_config=optimizer_config,
            activation_config=activation_config,
        )

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

            last_epoch_mean_train_mse = float(np.mean(epoch_losses))
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | mean train MSE: {last_epoch_mean_train_mse:.4f}")

        if verbose:
            print(f"\nCompleted fold {fold_idx+1}/{total_folds}.\n")

        # validation
        val_out = forward_pass(layers, validation_set)
        val_loss = float(mse.forward(val_out, validation_set))
        val_mses.append(val_loss)
        bottleneck_dim_out = bottleneck_dim
        if verbose:
            print(f"Validation MSE for fold {fold_idx+1}/{total_folds}: {val_loss:.4f}\n")

        # keep only last fold's reconstructions for final plotting
        last_validation = validation_set
        last_val_out = val_out

    # final single plot of reconstructions from the last fold
    if plot and last_validation is not None and last_val_out is not None:
        show_num = 5
        imgs = np.vstack((last_validation[:show_num], last_val_out[:show_num]))
        plot_gallery(
            "Final Reconstructions (last fold)",
            imgs,
            n_row=2,
            n_col=5,
        )

    mean_val_mse = float(np.mean(val_mses)) if len(val_mses) > 0 else None
    if bottleneck_dim_out is None:
        # fallback to last encoder size or input dim
        inferred_bottleneck = encoder_sizes[-1] if (encoder_sizes is not None and len(encoder_sizes) > 0) else n_features
        bottleneck_dim_out = inferred_bottleneck
    compression_ratio = float(n_features) / float(bottleneck_dim_out)

    return {
        "mean_train_mse_last_epoch": last_epoch_mean_train_mse,
        "mean_val_mse": mean_val_mse,
        "compression_ratio": compression_ratio,
        "bottleneck_dim": bottleneck_dim_out,
        "n_features": n_features,
    }

def main():
    # default settings retain previous behavior
    result = train_and_evaluate(
        encoder_sizes=[512, 128],
        epochs=50,
        batch_size=35,
        optimizer_config={
            "type": "adam",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 1e-5,
            "adamw": True,
        },
        activation_config={"type": "relu"},
        seed=0,
        plot=True,
        verbose=True,
    )
    print({"mean_val_mse": result["mean_val_mse"], "compression_ratio": result["compression_ratio"]})
if __name__ == "__main__":
    main()
