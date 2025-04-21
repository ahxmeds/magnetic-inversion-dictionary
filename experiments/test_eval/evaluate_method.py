import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ssmaginv.config import (METRICS_DIR, PREDS_DIR,
                             get_default_magnetics_config)
from ssmaginv.dataset.saved_dataset import get_test_set
from ssmaginv.solvers.dictionary import SingleDictRecovery, ScaleSpaceRecovery

from compute_tv_model import compute_tv_model
from compute_dict_model import get_inference_model_fn
from compute_cosine_model import compute_cosine_model


def compute_test_predictions(method: callable, name: str, batch_size: int = 16):
    """
    Compute predictions using a specified method (e.g., TV, L2) on the test dataset.

    Args:
        method (callable): The method to use for predictions.
        name (str): The name of the method for saving results.
        batch_size (int): The batch size for processing the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup precomputed dataset
    dataset = get_test_set()
    if dataset is None:
        raise RuntimeError("Test dataset is missing, compute the npz files first.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    save_dir = os.path.join(PREDS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.join(METRICS_DIR, name)
    os.makedirs(metrics_dir, exist_ok=True)

    # Store all losses
    x_losses, d_losses = [], []
    metrics = []

    test_set_index = 0
    for x_true, data_true, data_noise in dataloader:
        x_true = x_true.to(device)
        data_true = data_true.to(device)
        data_noise = data_noise.to(device)

        # Predict
        x_pred, data_pred = method(x_true, data_noise)

        # Determine batch size
        batch_dim = x_true.shape[0]

        # Compute per-sample normalized losses
        x_loss_dim = tuple(range(1, x_true.ndim))  # dims to reduce: (C, X, Y, Z)
        d_loss_dim = tuple(range(1, data_true.ndim))  # dims to reduce: (C, X, Y)

        # Compute per-sample losses
        batch_x_loss = torch.mean((x_pred - x_true) ** 2, dim=x_loss_dim) / torch.mean(
            x_true**2, dim=x_loss_dim
        )
        batch_d_loss = torch.mean(
            (data_pred - data_true) ** 2, dim=d_loss_dim
        ) / torch.mean(data_true**2, dim=d_loss_dim)

        # Save each sample’s results
        for i in range(batch_dim):
            sample_x_loss = batch_x_loss[i].item()
            sample_d_loss = batch_d_loss[i].item()
            x_losses.append(sample_x_loss)
            d_losses.append(sample_d_loss)

            metrics.append(
                {
                    "ImageID": f"{test_set_index:04d}",
                    "ModelLoss": sample_x_loss,
                    "DataLoss": sample_d_loss,
                }
            )

            # Save .npz with predicted + ground truth
            np.savez(
                os.path.join(save_dir, f"{test_set_index:04d}.npz"),
                x=x_pred[i].cpu().numpy(),
                x_true=x_true[i].cpu().numpy(),
                data=data_pred[i].cpu().numpy(),
                data_true=data_true[i].cpu().numpy(),
            )

            test_set_index += 1

    # Save losses to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(metrics_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run predictions on test set using a specified method."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["tv", "single_dict", "shared_dict", "cosine"],
        help="Inference method to use (currently only 'tv' is supported).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing the dataset",
    )
    args = parser.parse_args()

    # Select method
    if args.method == "tv":
        method = compute_tv_model
        name = "Variational"
        args.batch_size = 1 # Override for TV method since it thresholds
    elif args.method == "single_dict":
        method = get_inference_model_fn(
            model_type=SingleDictRecovery,
            saved_weights_path="SingleDict.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )    
        name = "SingleDict"
    elif args.method == "shared_dict":
        method = get_inference_model_fn(
            model_type=ScaleSpaceRecovery,
            saved_weights_path="ScaleSpace.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        name = "UnrolledDict"
    elif args.method == "cosine":
        # TODO: Add cosine transform method here function that take in x_true and data_noise
        # and returns x_pred and data_pred using cosine transform
        method = compute_cosine_model
        name = "CosineTransformDict"
        raise NotImplementedError("Cosine method is not implemented yet.")    
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    compute_test_predictions(method=method, name=name, batch_size=args.batch_size)
