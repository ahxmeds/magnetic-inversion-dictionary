import os, sys
import torch
import numpy as np
from tqdm import tqdm

import ssmaginv.config as config
from ssmaginv.dataset.mag_data import MagneticDataset
from ssmaginv.magnetics.magneticsFP import Magnetics
from torch.utils.data import DataLoader


def pad_zeros_at_front(num, N):
    return str(num).zfill(N)



def generate_dataset(name, dataset_config, save_dir, forward_model, device):
    """
    Generate a dataset (train, val, or test) and save it to .npz files.

    Parameters:
        name (str): Name of the dataset (for logging).
        dataset_config (dict): Configuration for the MagneticDataset.
        save_dir (str): Directory to save the .npz files.
        forward_model (Magnetics): Instance of the forward magnetic model.
    """
    total_samples = dataset_config['total_samples']
    print(f"Generating '{name}' set with {total_samples} samples...")

    dataset = MagneticDataset(**dataset_config)
    dataloader = DataLoader(dataset, batch_size=1)

    os.makedirs(save_dir, exist_ok=True)

    for i, x_true in enumerate(tqdm(dataloader, desc=f"Generating {name}", total=total_samples)):
        x_true = x_true.to(device)
        data_true = forward_model(x_true)
        noise_level = 0.01 * data_true.abs().mean()
        data_true_noise = data_true + noise_level * torch.randn_like(data_true)

        # Convert to numpy
        x_true_np = x_true[0, 0].cpu().numpy()
        data_true_np = data_true[0, 0].cpu().numpy()
        data_true_noise_np = data_true_noise[0, 0].cpu().numpy()

        save_dict = {
            "x_true": x_true_np,
            "data_true": data_true_np,
            "data_true_noise": data_true_noise_np,
        }

        file_name = f"{pad_zeros_at_front(i, 4)}.npz"
        np.savez(os.path.join(save_dir, file_name), **save_dict)

    
def reproduce_ss_paper_data():
    """
    Generate synthetic magnetic inversion datasets for train, validation, and test.
    Used to reproduce scale-space paper results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize forward magnetic model
    magnetics_config = config.get_default_magnetics_config(device=device)
    forward_model = Magnetics(**magnetics_config)

    # Common mesh size for all splits
    dim = config.DEFAULT_DIM

    # Dataset configurations
    dataset_configs = {
        "train": {
            "dim": dim,
            "num_ellipsoids_range": [1, 6],
            "seed": 42,
            "total_samples": 2000,
        },
        "valid": {
            "dim": dim,
            "num_ellipsoids_range": [1, 6],
            "seed": int(1e6),
            "total_samples": 100,
        },
        "test": {
            "dim": dim,
            "num_ellipsoids_range": [1, 6],
            "seed": int(1e8),
            "total_samples": 500,
        },
    }

    data_dirs = {
        "train": config.TRAIN_DIR,
        "valid": config.VALID_DIR,
        "test": config.TEST_DIR,
    }

    # Generate each dataset split
    for name in ["train", "valid", "test"]:
        generate_dataset(
            name=name,
            dataset_config=dataset_configs[name],
            save_dir=data_dirs[name],
            forward_model=forward_model,
            device=device,
        )
        
if __name__ == "__main__":
    reproduce_ss_paper_data()



