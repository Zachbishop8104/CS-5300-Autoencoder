import matplotlib.pyplot as plt

def plot_gallery(title, images, n_col=3, n_row=2, cmap=plt.cm.gray):
    image_shape = (64, 64)

    fig, axs = plt.subplots(
        nrows=n_row, ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white", constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)

    # display range fixed for Olivetti faces
    for ax, vec in zip(axs.flat, images):
        ax.imshow(vec.reshape(image_shape), cmap=cmap, interpolation="nearest",
                  vmin=0.0, vmax=1.0)
        ax.axis("off")

    plt.show()
