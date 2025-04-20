import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ssmaginv.config import TRAIN_DIR, TEST_DIR, VALID_DIR
from ssmaginv.dataset.generate_data import reproduce_ss_paper_data

class NpzMagneticDataset(Dataset):
    """
    Dataset wrapper for loading precomputed .npz magnetic inversion data.

    Each file is expected to contain:
        - x_true: ground truth model
        - data_true: forward model output
        - data_true_noise: noisy forward model output
    """

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.files = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".npz")
            ],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        x_true = torch.tensor(d["x_true"]).unsqueeze(0).float()
        data_true = torch.tensor(d["data_true"]).unsqueeze(0).float()
        data_noise = torch.tensor(d["data_true_noise"]).unsqueeze(0).float()
        return x_true, data_true, data_noise
    
def get_train_set():
    if not check_dataset_available(TRAIN_DIR):
        return None
    return NpzMagneticDataset(TRAIN_DIR)

def get_valid_set():
    if not check_dataset_available(VALID_DIR):
        return None
    return NpzMagneticDataset(VALID_DIR)

def get_test_set():
    if not check_dataset_available(TEST_DIR):
        return None
    return NpzMagneticDataset(TEST_DIR)

def check_dataset_available(directory):
    """
    Checks if the given dataset directory contains any .npz files.
    If not, warns the user to generate the dataset.
    """
    if not os.path.exists(directory) or not any(f.endswith(".npz") for f in os.listdir(directory)):
        print(f"[WARNING] Dataset directory '{directory}' is empty or missing.")
        print("To generate the dataset npz files first run the script `dataset/generate_data.py`")
        return False
    return True
    
if __name__ == "__main__":
    from ssmaginv.config import TRAIN_DIR
    from ssmaginv.plot import plot_model_with_forward
    
    dataset = NpzMagneticDataset(TRAIN_DIR)
    
    # Iterate through 3 samples and plot
    for i in range(3):
        x_true, data_true, data_noise = dataset[i]
        print(x_true.shape, data_noise.shape)
        plot_model_with_forward(x_true[0], data_noise[0]).show()