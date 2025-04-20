import os

import numpy as np
import pandas as pd
import pyvista as pv

from ssmaginv.config import METRICS_DIR, PREDS_DIR, PROJECT_ROOT
from ssmaginv.dataset.saved_dataset import get_test_set
from ssmaginv.plot import plot_model_with_forward


def get_model_pairs(idx):
    name = f"{idx:04d}.npz"
    # Retrive pairs from the PREDS dir
    pred_dir = PREDS_DIR
    methods = ["Variational", "CosineTransformDict", "SingleDict", "UnrolledDict"]
    pairs = []

    # Get the first pair (True model and data)
    dataset = get_test_set()
    x_true, data_true, data_noise = dataset[idx]
    pair = (x_true, data_true)
    pairs.append(pair)

    # Fetch the model predictions from each method
    for i, dir in enumerate(methods):
        dir = os.path.join(pred_dir, dir)
        # append the indexed value as a 4d format integer .npz to load
        file_path = os.path.join(dir, name)
        # Pull the dict keys from the npz file
        d = np.load(file_path)
        x = np.squeeze(d["x"])
        data = np.squeeze(d["data"])
        pair = (x, data)
        pairs.append(pair)

    return pairs


def get_model_losses(idx):
    metric_dir = METRICS_DIR
    methods = ["Variational", "CosineTransformDict", "SingleDict", "UnrolledDict"]

    losses = []
    losses.append((None, None))

    for i, dir in enumerate(methods):
        dir = os.path.join(metric_dir, dir)
        file_path = os.path.join(dir, "metrics.csv")

        data = pd.read_csv(file_path)
        x_ml_loss = data["ModelLoss"].to_numpy()
        d_ml_loss = data["DataLoss"].to_numpy()
        loss = (x_ml_loss[idx], d_ml_loss[idx])
        losses.append(loss)

    return losses


def plot_comparison_3d(idx=0):
    pairs = get_model_pairs(idx)
    losses = get_model_losses(idx)

    names = [
        "True Model",
        "Variational",
        "Cosine Dictionary",
        "Shared Dictionary",
        "Unrolled Dictionary",
    ]

    p = plot_data_pairs(pairs, names, losses, fig_size=(1800, 450))
    return p


def plot_data_pairs(
    data_pairs,
    names=None,
    losses=None,
    spacing=(1.0, 1.0, 1.0),
    height=0,
    fig_size=(1800, 500),
    clim_mag=None,
    clim_forward=None,
    title_font_size=12,
):
    # Validate input
    if not isinstance(data_pairs, (list, tuple)) or len(data_pairs) == 0:
        raise ValueError(
            "data_pairs must be a non-empty list or tuple of (mag_data, forward_data) pairs."
        )

    # Compute common clims if not provided
    if clim_mag is None:
        scalar_min_mag = min(pair[0].min() for pair in data_pairs).item()
        scalar_max_mag = max(pair[0].max() for pair in data_pairs).item()
        clim_mag = (scalar_min_mag, scalar_max_mag)

    if clim_forward is None:
        scalar_min_forward = min(pair[1].min() for pair in data_pairs).item()
        scalar_max_forward = max(pair[1].max() for pair in data_pairs).item()
        clim_forward = (scalar_min_forward, scalar_max_forward)

    # Initialize the PyVista Plotter with the desired layout
    num_pairs = len(data_pairs)
    p = pv.Plotter(
        shape=(1, num_pairs),  # Single row, one column per data pair
        window_size=fig_size,
    )

    for i, ((mag_data, forward_data), name) in enumerate(zip(data_pairs, names)):
        print(f"mag_data: {mag_data.shape}, forward_data: {forward_data.shape}")
        p.subplot(0, i)
        plot_model_with_forward(
            mag_data,
            forward_data,
            spacing,
            height,
            plotter=p,
            clim_mag=clim_mag,
            clim_forward=clim_forward,
        )
        if name is not None and name[i] is not None:
            p.add_title(name, font_size=title_font_size)
        if losses is not None and losses[i][0] is not None:
            loss_pair = losses[i]
            p.add_text(
                f"Loss m: {loss_pair[0]:.3f}", position="lower_left", font_size=12
            )
            p.add_text(
                f"Loss d: {loss_pair[1]:.4f}", position="lower_right", font_size=12
            )
            if i == 0:
                p.add_axes()
        p.zoom_camera(0.95)

    return p


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot 3D model comparison for a given sample index."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the test sample to visualize (default: 0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the plot to file instead of displaying it interactively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Relative directory to save the screenshot if --save is enabled",
    )

    args = parser.parse_args()

    if args.index >= 500:
        raise ValueError("Index must be less than 500 for test samples.")

    p = plot_comparison_3d(idx=args.index)

    if args.save:
        # join the output directory with the project root
        save_dir = os.path.join(PROJECT_ROOT, args.output_dir)
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"demo_img_{args.index:04d}.png")
        p.off_screen = True
        p.screenshot(img_path)
        print(f"Saved screenshot to {img_path}")
    else:
        p.show()
