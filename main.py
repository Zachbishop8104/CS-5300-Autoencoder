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
    trainable_layers = tuple([L for L in layers if hasattr(L, "weights")])

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

last_gallery_imgs = None


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
    val_size=10,
):
    # grab faces from selected seed
    rng = RandomState(seed)
    faces, targets = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng, data_home="data/olivetti_faces")
    n_samples, n_features = faces.shape
    subjects = np.unique(targets)
    groups = [faces[targets == s] for s in subjects]  # each is shape (10, 4096)

    mse = MSE()
    global last_gallery_imgs

    val_mses = []
    last_epoch_mean_train_mse = None
    bottleneck_dim_out = None

    # Build a small, fixed gallery set spanning multiple subjects.
    # This is the fallback if we cannot construct a subject-based
    # unseen/seen gallery (e.g., for non-subject CV modes).
    show_num = 5
    # pick up to show_num subjects spread across the dataset
    step = max(len(groups) // show_num, 1)
    subj_indices = [min(i * step, len(groups) - 1) for i in range(show_num)]
    gallery_inputs = np.vstack([groups[i][:1] for i in subj_indices])

    # Build folds:
    #   - LOSO (leave-one-subject-out) by subject
    #   - K-Fold by subject
    #   - Holdout by subject (single train/val split on unseen subjects)
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
    elif cv_mode == "holdout":
        # Single train/validation split at the SUBJECT level (unseen people).
        # We interpret val_size as "approximately this many images" and,
        # assuming ~10 images per subject, pick enough WHOLE subjects to
        # reach about that many images. We then take the *final* subjects
        # (highest indices) as the validation set so the split is stable.
        total_folds = 1
        m = len(groups)  # number of subjects

        vs_images = int(val_size) if val_size is not None else 20
        if vs_images <= 0:
            vs_images = 20
        imgs_per_subject = groups[0].shape[0] if m > 0 else 10
        n_val_subjects = max(1, vs_images // max(1, imgs_per_subject))
        if n_val_subjects >= m:
            n_val_subjects = max(1, m - 1)

        # Use the last n_val_subjects as validation ("final" subjects)
        val_idx = indices[-n_val_subjects:]
        train_idx = indices[:-n_val_subjects]

        def _holdout_iter():
            yield (train_idx, val_idx)
        fold_iter = _holdout_iter()
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

    layers_for_gallery = None
    # Subject indices for the fold whose model we keep for visualization
    val_subject_idx_for_gallery = None
    train_subject_idx_for_gallery = None

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
        # Build train/val sets depending on CV mode
        if cv_mode in ("kfold", "loso", "holdout"):
            # stack subjects into train/val arrays (unseen people in val set)
            training_set = np.vstack([groups[i] for i in train_idx])
            validation_set = np.vstack([groups[i] for i in val_idx])
            # For subject-based CV modes we want the gallery to show
            # reconstructions of people the model WAS and WAS NOT trained on.
            # Keep track of the subject indices for the (last) fold.
            val_subject_idx_for_gallery = np.array(val_idx, copy=True)
            train_subject_idx_for_gallery = np.array(train_idx, copy=True)
        else:
            # fallback: split directly over individual images
            training_set = faces[train_idx]
            validation_set = faces[val_idx]

        layers, opt, bottleneck_dim = initialize_model(
            num_features=n_features,
            encoder_sizes=encoder_sizes,
            optimizer_config=optimizer_config,
            activation_config=activation_config,
        )

        # validate every fixed number of epochs so we can watch for overfitting
        val_interval = 10

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

            # Periodic validation to monitor overfitting
            val_info = ""
            if ((epoch + 1) % val_interval == 0) or ((epoch + 1) == epochs):
                val_out_epoch = forward_pass(layers, validation_set)
                val_loss_epoch = float(mse.forward(val_out_epoch, validation_set))
                val_info = f" | val MSE: {val_loss_epoch:.4f}"

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | mean train MSE: {last_epoch_mean_train_mse:.4f}{val_info}")

        if verbose:
            print(f"\nCompleted fold {fold_idx+1}/{total_folds}.\n")

        # validation
        val_out = forward_pass(layers, validation_set)
        val_loss = float(mse.forward(val_out, validation_set))
        val_mses.append(val_loss)
        bottleneck_dim_out = bottleneck_dim
        # keep last-trained model for final gallery visualization
        layers_for_gallery = layers
        if verbose:
            print(f"Validation MSE for fold {fold_idx+1}/{total_folds}: {val_loss:.4f}\n")

    # final plot of reconstructions on a gallery set
    imgs = None
    n_row = 0
    n_col = 0

    if layers_for_gallery is not None:
        unseen_inputs = None
        seen_inputs = None

        if val_subject_idx_for_gallery is not None and train_subject_idx_for_gallery is not None:
            # Build unseen (validation) inputs
            val_subjects = list(val_subject_idx_for_gallery)
            train_subjects = list(train_subject_idx_for_gallery)

            if cv_mode == "holdout" and len(val_subjects) == 2:
                # Special case: final run with two unseen people.
                s0, s1 = val_subjects[0], val_subjects[1]
                g0 = groups[s0]
                g1 = groups[s1]
                n0 = min(3, g0.shape[0])
                n1 = min(show_num - n0, g1.shape[0])
                unseen_parts = [g0[:n0]]
                if n1 > 0:
                    unseen_parts.append(g1[:n1])
                unseen_inputs = np.vstack(unseen_parts)
            else:
                # Generic subject-based CV: up to show_num unseen people
                chosen_val = val_subjects[:show_num]
                unseen_inputs = np.vstack([groups[s][:1] for s in chosen_val])

            # Build seen (training) inputs: up to show_num different people
            if len(train_subjects) >= show_num:
                chosen_train = train_subjects[:show_num]
                seen_inputs = np.vstack([groups[s][:1] for s in chosen_train])
            elif len(train_subjects) > 0:
                # Fallback: repeat some subjects if there are fewer than show_num
                tiles = []
                for i in range(show_num):
                    s = train_subjects[i % len(train_subjects)]
                    tiles.append(groups[s][:1])
                seen_inputs = np.vstack(tiles)

        if unseen_inputs is not None and seen_inputs is not None:
            # Compose gallery with four rows:
            # 1) unseen originals, 2) unseen reconstructions,
            # 3) seen originals,   4) seen reconstructions.
            unseen_out = forward_pass(layers_for_gallery, unseen_inputs)
            seen_out = forward_pass(layers_for_gallery, seen_inputs)
            imgs = np.vstack((unseen_inputs, unseen_out, seen_inputs, seen_out))
            n_row = 4
            n_col = unseen_inputs.shape[0]
        elif gallery_inputs is not None:
            # Fallback: original behaviour (mixed fixed gallery)
            gallery_out = forward_pass(layers_for_gallery, gallery_inputs)
            imgs = np.vstack((gallery_inputs[:show_num], gallery_out[:show_num]))
            n_row = 2
            n_col = show_num

    if imgs is not None:
        # expose gallery images for external callers (e.g., run_test)
        last_gallery_imgs = imgs

        if plot and n_row > 0 and n_col > 0:
            plot_gallery(
                "Final Reconstructions (gallery set)",
                imgs,
                n_row=n_row,
                n_col=n_col,
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
