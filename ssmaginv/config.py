import os
import torch
from numpy import pi as np_pi

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default physical model parameters
DEFAULT_DIM = torch.tensor([64, 64, 32])  # Grid dimensions
DEFAULT_CELL_SIZE = torch.tensor([100.0, 100.0, 100.0])  # Cell sizes in meters
DEFAULT_DIRS = torch.tensor([np_pi / 2] * 4)  # Magnetic angles (I, A, I0, A0)

def get_default_magnetics_config(device="cpu"):
    return {
        "dim": DEFAULT_DIM,
        "h": DEFAULT_CELL_SIZE,
        "dirs": DEFAULT_DIRS,
        "device": torch.device(device),
    }

# Base data directory (relative to repo or via env override)
DATA_ROOT = os.getenv("SSMAGINV_DATA", os.path.join(PROJECT_ROOT, "data"))

TEST_DIR = os.path.join(DATA_ROOT, "testset")
TRAIN_DIR = os.path.join(DATA_ROOT, "trainset")
VALID_DIR = os.path.join(DATA_ROOT, "validset")

for d in [TEST_DIR, TRAIN_DIR, VALID_DIR]:
    os.makedirs(d, exist_ok=True)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "RESULTS")
METRICS_DIR = os.path.join(RESULTS_DIR, "METRICS")
PREDS_DIR = os.path.join(RESULTS_DIR, "PREDS")

for d in [RESULTS_DIR, METRICS_DIR, PREDS_DIR]:
    os.makedirs(d, exist_ok=True)
    
    