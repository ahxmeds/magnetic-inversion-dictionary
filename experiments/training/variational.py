import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

import ssmaginv.config as config
from ssmaginv.magnetics import Magnetics
from ssmaginv.dataset.mag_data import MagneticDataset
from ssmaginv.dataset.saved_dataset import get_train_set, get_valid_set, get_test_set
from ssmaginv.solvers.variational import solve_tv, set_z_weights
from ssmaginv.plot.plot import plot_mixed

    
def grid_search(output_path="grid_search_results.csv"):
    """
    Perform a grid search over the parameters alpha, beta, and z0 to find the best combination
    for the model inversion task.
    
    Hyperparameters:
        - alpha: TV regularization parameter
        - beta: Z-weighting regularization parameter
        - z0: Z-weighting parameter
        - w1: Weight for data loss |data_true - data_pred|^2
        - w2: Weight for model loss |x_true - x_pred|^2
        
    The results are saved to a CSV file for further analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup magnetics model size and operator
    model_cfg = config.get_default_magnetics_config(device=device)
    dim = model_cfg["dim"]
    h = model_cfg["h"]
    forMod = Magnetics(**model_cfg)

    # Setup precomputed dataset
    dataset = get_valid_set()
    if dataset is None:
        raise RuntimeError("Dataset is missing.")
    
    # Define the DataLoader
    batch_size = 16
    num_workers = 0  
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Gid search hyperparameters (coarse search)
    # alpha_values = np.logspace(-5, -1, num=5)  # Adjust the range and number of points
    # beta_values = np.logspace(-5, -1, num=5)
    # z0_values = np.linspace(0.1, 1.0, num=3)

    # Grid search hyperparameters (fine-tune search)
    alpha_values = np.array(
        [
            0.5e-4,
            1e-4,
            2e-4,
        ]
    )  
    beta_values = np.array(
        [
            0.5e3,
            1e3,
            2e3,
        ]
    )
    z0_values = np.array([5, 10, 20])

    w1, w2 = 5.0, 1.0

    best_loss = float("inf")
    best_params = None

    results = []

    for alpha_val in alpha_values:
        for beta_val in beta_values:
            for z0_val in z0_values:

                # Create a z-weighting tensor
                zdim = dim[-1]
                zscale = h[-1]
                Z_weights = set_z_weights(zdim, zscale, z0_val)

                total_data_loss = 0.0
                total_model_loss = 0.0
                total_loss = 0.0
                for xtrue, data_true, data_noise in data_loader:
                    xtrue = xtrue.to(device)
                    data_true = data_true.to(device)
                    D = data_noise.to(device)
                    x0 = torch.zeros_like(xtrue)

                    xinv = solve_tv(
                        x0,
                        dim,
                        D,
                        forMod,
                        alpha=alpha_val,
                        Z_weights=Z_weights,
                        beta=beta_val,
                        n_iters=12,
                    )

                    # Compute losses
                    data_loss = torch.mean((data_true - forMod(xinv)) ** 2) / torch.mean(D**2)
                    model_loss = torch.mean((xinv - xtrue) ** 2) / torch.mean(xtrue**2)
                    loss = w1 * data_loss + w2 * model_loss

                    total_data_loss += data_loss.item()
                    total_model_loss += model_loss.item()
                    total_loss += loss.item()

                # Calculate averages over all batches
                num_batches = len(data_loader)
                avg_data_loss = total_data_loss / num_batches
                avg_model_loss = total_model_loss / num_batches
                avg_loss = total_loss / num_batches

                # Store the results
                results.append(
                    {
                        "alpha": alpha_val,
                        "beta": beta_val,
                        "z0": z0_val,
                        "avg_loss": avg_loss,
                        "avg_data_loss": avg_data_loss,
                        "avg_model_loss": avg_model_loss,
                    }
                )

                # Print progress
                print(
                    f"Alpha: {alpha_val:.1e}, Beta: {beta_val:.1e}, z0: {z0_val:.2f}, "
                    f"Avg Loss: {avg_loss:.6e}, Avg Data Loss: {avg_data_loss:.6e}, "
                    f"Avg Model Loss: {avg_model_loss:.6e}"
                )

                # Update best parameters
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_params = (alpha_val, beta_val, z0_val)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_path, index=False)

    print(
        f"\nBest Parameters - Alpha: {best_params[0]}, Beta: {best_params[1]}, z0: {best_params[2]}"
    )

def solve_single(idx):
    """
    Demonstration of solving a single inversion problem using the variational method.
    """
    # Random seed for reproducibility
    RAND_SEED = 100

    dim = torch.tensor([64, 64, 32])
    h = torch.tensor([100.0, 100.0, 100.0])
    dirs = torch.tensor([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]) * 1.0
    forMod = Magnetics(dim, h, dirs)

    n1, n2, n3 = dim

    dataset = MagneticDataset(
        dim=dim,
        num_ellipsoids_range=[1, 6],
        seed=RAND_SEED,  # Dataset seed for reproducibility
        total_samples=128,  # Number of samples per epoch
    )

    # Generate a forward sample
    xtrue = dataset[idx]
    xtrue = xtrue.unsqueeze(0)
    D = forMod(xtrue)
    noise = torch.randn_like(D)
    D = D + 0.01 * noise

    # Initial guess
    x0 = torch.zeros_like(xtrue)

    # Invert back
    # xinv = solve_ls(x0, D, forMod, reg_func=reg_func, alpha=alpha)
    alpha = 1e-4  # TV regularization
    beta = 1e3  # Z-weighting regularization
    z0 = 10  # Z-weighting parameter

    # Create a z-weighting tensor
    Z_weights = set_z_weights(dim[-1], h[-1], z0)
    xinv = solve_tv(
        x0,
        dim,
        D,
        forMod,
        alpha=alpha,
        Z_weights=Z_weights,
        beta=beta,
        n_iters=12,
        verbose=True,
    )

    # Prediced forward model
    pred = forMod(xinv)  # predicted 2d data
    lossd = torch.mean((D - pred) ** 2) / torch.mean((D) ** 2)
    lossx = torch.mean((xinv - xtrue) ** 2) / torch.mean((xtrue) ** 2)
    print(f"Data Loss: {lossd:.6e}")
    print(f"Model Loss: {lossx:.6e}")
    print(f"Total Loss: {(lossd+lossx):.6e}")

    plt.subplot(1, 2, 1)
    plt.imshow(D.view(n1, n2).cpu().detach().numpy(), cmap="rainbow")
    plt.title("Data")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred.view(n1, n2).cpu().detach().numpy(), cmap="rainbow")
    plt.title("Inverted model")
    plt.colorbar()

    p = plot_mixed(xtrue, xinv, D, pred)
    p.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run variational grid search.")
    parser.add_argument(
        "--output",
        type=str,
        default="grid_search_results.csv",
        help="Path to save the CSV file",
    )
    args = parser.parse_args()

    grid_search(output_path=args.output)