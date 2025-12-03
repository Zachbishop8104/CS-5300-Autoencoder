import sys
import json
import os
import time

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as ae_main
from gallary import plot_gallery


def _build_optimizer_config(cfg):
    opt = cfg.get("optimizer")
    if not isinstance(opt, dict) or not opt.get("type"):
        raise ValueError("optimizer config with 'type' is required (adam or sgd)")
    out = opt.copy()
    out["type"] = str(out.get("type", "adam")).lower()
    return out


def _build_activation_config(cfg):
    act = cfg.get("activation", {"type": "relu"})
    if not isinstance(act, dict) or not act.get("type"):
        raise ValueError("activation config with 'type' is required (relu or leaky_relu)")
    out = act.copy()
    out["type"] = str(out.get("type", "relu")).lower()
    return out


_GALLERY_INDEX = 0


def _run_single(cfg):
    encoder_sizes = cfg.get("encoder_sizes", [512, 128])
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 35))
    seed = int(cfg.get("seed", 0))
    plot = bool(cfg.get("plot", False))
    verbose = bool(cfg.get("verbose", False))
    cv_type = str(cfg.get("cv_type", "loso")).lower()
    n_splits = int(cfg.get("n_splits", 5))
    val_size = int(cfg.get("val_size", 10))
    optimizer_config = _build_optimizer_config(cfg)
    activation_config = _build_activation_config(cfg)

    t0 = time.perf_counter()
    result = ae_main.train_and_evaluate(
        encoder_sizes=encoder_sizes,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_config=optimizer_config,
        seed=seed,
        plot=plot,
        verbose=verbose,
        activation_config=activation_config,
        cv_type=cv_type,
        n_splits=n_splits,
        val_size=val_size,
    )
    t1 = time.perf_counter()
    train_time_s = float(t1 - t0)

    # attach context for reporting
    result_out = {
        "encoder_sizes": encoder_sizes,
        "epochs": epochs,
        "batch_size": batch_size,
        "cv_type": cv_type,
        "n_splits": n_splits,
        "optimizer": optimizer_config,
        "activation": activation_config,
        "train_time_s": train_time_s,
        **result,
    }

    # Generate and save a reconstruction gallery for this network, if available
    try:
        imgs = getattr(ae_main, "last_gallery_imgs", None)
        if imgs is not None:
            global _GALLERY_INDEX
            mse = result.get("mean_val_mse")
            mse_str = f", mse={mse:.4f}" if mse is not None else ""
            title = (
                f"Reconstructions (network {_GALLERY_INDEX + 1})\n"
                f"enc={encoder_sizes}, opt={optimizer_config.get('type')}, "
                f"act={activation_config.get('type')}{mse_str}"
            )
            gallery_filename = f"gallery_network_{_GALLERY_INDEX + 1}.png"
            # Infer grid size from the number of images (we always use 5 columns).
            n_col = 5
            n_row = max(1, int((len(imgs) + n_col - 1) // n_col))
            plot_gallery(
                title,
                imgs,
                n_row=n_row,
                n_col=n_col,
                save_path=gallery_filename,
            )
            _GALLERY_INDEX += 1
    except Exception:
        # Do not fail the run if gallery plotting fails
        pass

    return result_out


def main():
    # Read JSON from file path arg or stdin
    data = None
    input_label = "stdin"
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        input_label = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        raw = sys.stdin.read()
        data = json.loads(raw)

    # Allow single config object or list of configs
    if isinstance(data, dict):
        results = [_run_single(data)]
    elif isinstance(data, list):
        results = [_run_single(cfg) for cfg in data]
    else:
        raise ValueError("Input JSON must be an object or array of objects")

    # Print machine-readable results
    print(json.dumps(results, indent=2))

    # After printing JSON, save training time vs accuracy (raw MSE) scatter
    try:
        xs = []  # training time (s)
        ys = []  # mean validation MSE
        labels = []
        for r in results:
            tt = r.get("train_time_s")
            mse = r.get("mean_val_mse")
            if tt is None or mse is None:
                continue
            xs.append(float(tt))
            ys.append(float(mse))
            enc = r.get("encoder_sizes")
            opt = r.get("optimizer", {})
            act = r.get("activation", {})
            label = f"enc={enc}, opt={opt.get('type')}, act={act.get('type')}"
            labels.append(label)

        if len(xs) > 0:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(xs, ys, c="#1f77b4")
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel("Training time (s)")
            ax.set_ylabel("Validation MSE (lower is better)")
            ax.set_title(f"{input_label}: Training Time vs Validation MSE")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            outfile = f"{input_label}.png"
            plt.savefig(outfile, dpi=150)
            plt.close(fig)
    except Exception as e:
        # Do not fail the run if plotting fails
        sys.stderr.write(f"Warning: failed to save compression_vs_accuracy.png: {e}\n")


if __name__ == "__main__":
    main()


