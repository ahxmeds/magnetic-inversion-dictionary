"""
Implementation of scale space dictionary learning for the magnetic inverse problem.

References:
[1] Ahamed, Shadab, et al. "Inversion of Magnetic Data using Learned Dictionaries and Scale Space."
arXiv preprint arXiv:2502.05451 (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleSpaceRecovery(nn.Module):
    """
    Equation (6) from [1].
    This class implements the scale space recovery algorithm for magnetic inversion.
    
    Parameters:
        forOp (object): Forward operator for the magnetic inversion problem.
        hid_dim (int): Hidden dimension channels for latent space representation.
        niter (int): Number of soft thresholding iterations.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Device to run the model on.
    """

    def __init__(
        self, forOp, hid_dim=9, niter=16, kernel_size=7, device=torch.device("cuda:0")
    ):
        super(ScaleSpaceRecovery, self).__init__()
        self.forOp = forOp
        self.device = device
        self.kernel_size = kernel_size
        ks = self.kernel_size
        self.niter = niter
        self.W = nn.Parameter(
            torch.randn(niter, 1, hid_dim, ks, ks, ks).to(self.device)
        )
        self.mu_j, self.tau = 1e-3, 2e-5
        self.pad = kernel_size // 2
        self.soft_threshold = lambda x, tau: torch.sign(x) * torch.clamp(x.abs() - tau, min=0)
        
    def _dict(self, i: int):
        """Return the 3‑D kernel to use at iteration *i* (sub‑classes override)."""
        return self.W[i]

    def forward(self, d):
        d = d.to(self.device)
        # Initial step is with z=0, simplifies the first iteration
        ATb = self.forOp.adjoint(d)
        dz = F.conv_transpose3d(ATb, self._dict(0), stride=1, padding=self.pad)
        z = torch.zeros_like(dz)

        for i in range(self.niter):
            z = self.soft_threshold(z - self.mu_j * dz, self.tau)

            # r = A * Psi * z^j - d
            x = F.conv3d(z, self._dict(i), padding=self.pad)
            Ax = self.forOp(x)
            r = Ax - d

            # Psi^T * A^T * r
            ATr = self.forOp.adjoint(r)
            dz = F.conv_transpose3d(ATr, self._dict(i), stride = 1, padding=self.pad)

        # Final threshold and sparse latent representation
        z = self.soft_threshold(z - self.mu_j * dz, self.tau)

        # Convert sparse latent representation to magnetic model space
        x = F.conv3d(z, self._dict(self.niter - 1), padding=self.pad)

        # Compute forward model prediction
        Ax = self.forOp(x)
        return z, x, Ax


class SingleDictRecovery(ScaleSpaceRecovery):
    """
    Single Shared Dictionary Recovery.
    """
    def __init__(self, forOp, hid_dim=9, niter=16, kernel_size=7,
                 device=torch.device("cuda:0")):
        super().__init__(forOp, hid_dim, niter, kernel_size, device)
       
        # Replace iterative kernel with a single shared kernel
        self.W = nn.Parameter(
            torch.randn(1, hid_dim, kernel_size, kernel_size, kernel_size,
                        device=device)
        )

    def _dict(self, i: int):
        """Ignore *i*: always return the shared kernel."""
        return self.W
