import numpy as np
import torch
from torch.utils.data import Dataset


class MagneticDataset(Dataset):
    """
    PyTorch Dataset for generating synthetic 3D magnetic models with ellipsoids.

    Parameters
    ----------
    dim : tuple
        Dimensions of the 3D grid for the model (n1, n2, n3).
    num_ellipsoids_range : list, optional
        Range (inclusive) of the random number of ellipsoids to include in each model.
    total_samples : int, optional
        Total number of samples to generate in the dataset.
    seed : int, optional
        Global seed for the dataset to ensure reproducibility.
    """

    def __init__(
        self,
        dim,
        num_ellipsoids_range=[1, 6],
        seed=None,
        total_samples=1000,
    ):
        self.dim = dim
        self.n1, self.n2, self.n3 = dim
        self.num_ellipsoids_range = num_ellipsoids_range
        self.centers_range = [0.2, 0.8]
        self.seed = seed
        self.total_samples = total_samples

        # Generate grids once during initialization
        self.x_grid, self.y_grid, self.z_grid = self.generate_torch_grids()

    def generate_torch_grids(self):
        x = torch.linspace(0, 1, self.n1)
        y = torch.linspace(0, 1, self.n2)
        z = torch.linspace(0, 1, self.n3)
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")
        return x_grid, y_grid, z_grid

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Derive a deterministic seed for this sample
        sample_seed = (self.seed + idx) if self.seed is not None else None

        # Set the torch generator for the sample
        tgen = torch.Generator(device="cpu")
        tgen.manual_seed(sample_seed)

        # Generate the magnetic model
        mag_model = self.generate_mag_model(gen=tgen)

        return mag_model

    def generate_mag_model(self, gen):
        """
        Generate ellipsoids and return their summed contribution to the magnetic model.
        """
        # Random number of ellipsoids
        num_ellipsoids = torch.randint(
            self.num_ellipsoids_range[0],
            self.num_ellipsoids_range[1] + 1,
            (1,),
            generator=gen,
        )

        # Random centers between 0.2 and 0.8
        centers = self.centers_range[0] + (
            self.centers_range[1] - self.centers_range[0]
        ) * torch.rand(num_ellipsoids, 3, generator=gen)
        # Generate magnetic susceptibility values for each ellipsoid
        mag_values = self.generate_mag_suscept_vals(num_ellipsoids, gen=gen)

        def get_axis_scale():
            """Get scalings for the ellipsoids, set to 1 for now"""
            return torch.ones(num_ellipsoids, 1)

        a, b, c = get_axis_scale(), get_axis_scale(), get_axis_scale()

        e = self.ellipsoid3d(centers, a, b, c)

        mag_values = mag_values.view(-1, 1, 1, 1)
        e = e * mag_values
        e_sum = e.sum(dim=0)  # Sum over all ellipsoids

        mag_model = e_sum.unsqueeze(0)  # Add channel dimension

        return mag_model

    def generate_mag_suscept_vals(self, n_vals=1, gen=None):
        """Generate magnetic susceptibility values for features"""
        return torch.rand(n_vals, generator=gen)

    def ellipsoid3d(self, centers, a, b, c, t=50):
        """
        Generate a batch of 3D ellipsoids with an exponential drop-off at edges.
        """
        x0 = centers[:, 0].view(-1, 1, 1, 1)
        y0 = centers[:, 1].view(-1, 1, 1, 1)
        z0 = centers[:, 2].view(-1, 1, 1, 1)
        a = a.view(-1, 1, 1, 1)
        b = b.view(-1, 1, 1, 1)
        c = c.view(-1, 1, 1, 1)
        e = (
            ((self.x_grid.unsqueeze(0) - x0) ** 2) / a**2
            + ((self.y_grid.unsqueeze(0) - y0) ** 2) / b**2
            + ((self.z_grid.unsqueeze(0) - z0) ** 2) / c**2
        )
        func_of_R = torch.exp(-t * e)
        return func_of_R  # Shape: (num_ellipsoids, n1, n2, n3)
