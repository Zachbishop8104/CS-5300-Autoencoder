## CS-5300 Autoencoder - Project Documentation

This project implements a from-scratch fully-connected autoencoder in NumPy to reconstruct faces from the Olivetti dataset. It includes a minimal training loop, modular layer primitives, and plotting utilities for visualizing reconstructions.

### Overview
- **Goal**: Learn a compressed representation of 64×64 grayscale face images and reconstruct them with low mean squared error (MSE).
- **Dataset**: Olivetti faces (400 images, 40 subjects × 10 images each). Loaded via `sklearn.datasets.fetch_olivetti_faces` into `data/olivetti_faces`.
- **Approach**: A symmetric encoder–decoder MLP trained with MSE using either Adam or SGD optimizers.
- **Outputs**: Per-epoch training MSE, per-fold validation MSE, and visual grids comparing originals vs reconstructions.

### Project Structure
- `main.py`: Entry point; defines model, training/validation loop, K-fold-by-subject split, and visualization.
- `gallary.py`: Helper to plot grids of images (`plot_gallery`).
- `meanSquaredError.py`: MSE loss with forward/backward.
- `layers/`
  - `linearLayer.py`: Fully-connected affine layer with He/Xavier init and gradients.
  - `reluLayer.py`: ReLU activation.
  - `sigmoidLayer.py`: Sigmoid activation for output layer.
  - `identityLayer.py`: Pass-through (not used by default).
- `data/olivetti_faces/`: Dataset cache folder created by scikit-learn.
- `requirements.txt`: Python dependencies.

Note: `main.py` also references `leakyReluLayer` and `optimizers` modules (`Adam`, `SGD`). Ensure these exist in your environment; otherwise replace `leakyReluLayer` with `reluLayer` and remove optimizer imports or add the missing modules.

### Data and Preprocessing
- Images are loaded as row vectors of length 4096 (64×64) scaled to [0,1].
- The code shuffles all 400 images, identifies unique subject IDs, and groups the 10 images per subject.
- Cross-validation is performed by leaving one subject (10 images) out as a validation fold (40 folds in total).

### Model Architecture
Defined in `initialize_model` within `main.py`:
- Encoder:
  - `Linear(4096 → 512)` → `LeakyReLU`
  - `Linear(512 → 128)` → `LeakyReLU`
- Decoder:
  - `Linear(128 → 512)` → `LeakyReLU`
  - `Linear(512 → 4096, init=xavier)` → `Sigmoid`

Implementation details:
- A small positive bias (`+0.01`) is added to all linear layers to encourage early activation.
- Weight initialization: He by default, Xavier for the final layer for stability with sigmoid.
- Output activation is sigmoid to match the [0,1] pixel range; loss is MSE.

### Training Loop
For each fold:
1. Instantiate layers and optimizer (Adam by default; SGD optional).
2. For each epoch (default 50):
   - Shuffle training set and iterate in mini-batches (default 35).
   - Forward pass through all layers.
   - Compute MSE loss and its gradient.
   - Backpropagate gradients layer-by-layer.
   - Call `opt.step()` to update parameters.
   - Track and print mean training loss.
3. After training, evaluate on the held-out subject and print validation MSE.
4. Visualize original vs reconstructed images for a subset.

### Modules and APIs
- Loss: `meanSquaredError.MSE`
  - `forward(pred, target) -> float`: Returns batch-mean MSE scaled by feature dimension.
  - `backward() -> grad`: Returns gradient w.r.t. predictions with shape `(B, D)`.

- Layers:
  - `linearLayer(in_features, out_features, weight_initialize_type="he"|"xavier"|other)`
    - `forward(X) -> Y`
    - `backward(dY) -> dX` and stores `dW`, `db` for optimizer use.
  - `reluLayer`
    - `forward(X) -> ReLU(X)`; `backward(dY) -> dY * (X>0)`.
  - `sigmoidLayer`
    - `forward(X) -> sigmoid(X)` with clipping for numerical stability; `backward(dY)` uses cached output.
  - `identityLayer`
    - Pass-through useful for debugging or architecture variants.

- Visualization: `gallary.plot_gallery(title, images, n_col, n_row, cmap)`
  - Expects `images` as a batch of flattened vectors; reshapes to 64×64 and displays.

### Optimizers (Expected Interfaces)
`main.py` expects optimizers providing `step()` and initialized with a tuple of trainable layers. If you add/modify optimizers, ensure they:
- Hold references to layers with `weights`, `bias`, `dW`, `db`.
- Implement learning-rate scheduling and optional weight decay as needed.

### Extending the Project
- Swap activations: Replace `leakyReluLayer` with `reluLayer` or add new activations.
- Change bottleneck size: Adjust encoder/decoder layer widths.
- Try different losses (e.g., MAE) or regularization (weight decay, dropout).
- Add checkpoints: Save `weights` and `bias` arrays between epochs/folds.
- Add metrics: PSNR or SSIM for reconstruction quality.

### Running and Usage
Assuming dependencies are installed (see `requirements.txt`):
- Run `main.py`. It will download/cache the dataset on first run under `data/olivetti_faces/`.
- Training prints epoch-wise mean training MSE and fold validation MSE.
- After each fold, a window displays original and reconstructed faces.

Configuration knobs (edit in `main.py`):
- `epochs` (default 50)
- `batch_size` (default 35)
- `use_adam` and `learning_rate` in `initialize_model`

### Testing Multiple Configurations (test.py)
Use `test.py` to run one or more experiment configurations specified in JSON. It calls `main.train_and_evaluate` and prints a JSON array of results with mean validation MSE and compression ratio.

Input JSON can be either a single object or an array of objects. Supported fields:
- `encoder_sizes`: list of integers (e.g., `[512, 128]`). Defines encoder hidden sizes; decoder mirrors it.
- `epochs`: integer.
- `batch_size`: integer.
- `seed`: integer (default 0).
- `plot`: boolean (default false) to enable reconstruction plots during testing.
- `verbose`: boolean (default false).
- `optimizer`: object with:
  - `type`: `"adam"` or `"sgd"`.
  - For Adam: `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `adamw`.
  - For SGD: `lr`.

Back-compat fields (optional):
- `use_adam`: boolean. If present, selects Adam/SGD and reads nested `adam` or `sgd` objects.
- `learning_rate`: number used if specific optimizer lr not provided.

Example single run:
```
{
  "encoder_sizes": [512, 128],
  "epochs": 20,
  "batch_size": 35,
  "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 1e-5}
}
```

Example multiple runs:
```
[
  {"encoder_sizes": [512, 128], "epochs": 10, "batch_size": 35, "optimizer": {"type": "adam", "lr": 0.001}},
  {"encoder_sizes": [256, 64], "epochs": 10, "batch_size": 50, "optimizer": {"type": "sgd", "lr": 0.01}}
]
```

Output fields per run:
- `mean_train_mse_last_epoch`
- `mean_val_mse`
- `compression_ratio` (input_dim / bottleneck_dim)
- `bottleneck_dim`
- `n_features`
- echo of `encoder_sizes`, `epochs`, `batch_size`, and `optimizer`

### Troubleshooting
- Missing modules: If you see import errors for `leakyReluLayer` or `optimizers`, either add those files or switch to available layers/optimizers.
- Empty plots or shape errors: Ensure images are `(B, 4096)` and `plot_gallery` receives `n_row * n_col` images.
- Numerical stability: Final layer uses Xavier init and sigmoid with clipping to handle large magnitudes; consider reducing learning rate if loss explodes.

### References
- scikit-learn Olivetti example linked in `README.md`.

---
Maintainers: Update this document whenever you change model architecture, training regime, or add new modules (e.g., optimizers, activations).


