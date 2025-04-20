# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class Magnetics(nn.Module):
    """
    A PyTorch module to perform forward and adjoint computations for magnetic inversion problems.
    """

    def __init__(self, dim, h, dirs, device="cpu"):
        """
        Initialize the Magnetics module.

        Parameters:
            dim (list or tuple): Mesh dimensions [nx, ny, nz] (meters).
            h (list or tuple): Cell sizes [dx, dy, dz] (meters).
            dirs (list or tuple): Magnetic field directions [I, A, I0, A0] in radians.
                                  I  - Magnetization dip angle
                                  A  - Magnetization declination angle
                                  I0 - Geomagnetic dip angle
                                  A0 - Geomagnetic declination angle
            device (str): Device to perform computations ('cpu' or 'cuda').
        """
        super(Magnetics, self).__init__()
        self.dim = dim
        self.h = h
        self.dirs = dirs
        self.device = device
        
        # Compute the scale factor for the magnetic field response
        dV = torch.prod(self.h)
        mu_0 = 1
        zeta = mu_0 / (4 * np.pi)
        self.mudV = zeta * dV

    def fft_kernel(self, P, center):
        """
        Compute the 2D shifted FFT of the kernel P.

        Parameters:
            P (torch.Tensor): The point spread function (PSF) kernel.
            center (list): Center indices for fftshift.

        Returns:
            torch.Tensor: The shifted FFT of P.
        """
        # use centers and the response and shift the data for FD operations
        S = torch.fft.fftshift(torch.roll(P, shifts=center, dims=[0, 1]))
        # take the fft
        S = torch.fft.fft2(S)
        # shift again to swap quadrants
        S = torch.fft.fftshift(S)
        return S

    def forward(self, M, height = 0):
        """
        Perform the forward computation using FFT.

        Parameters:
            M (torch.Tensor): The magnetization model tensor of shape B,C,X,Y,Z.

        Returns:
            torch.Tensor: The computed magnetic data.
        """
        dz = self.h[2]
        z = height + dz / 2

        data = 0

        # Loop through each layer in the z-direction
        for i in range(M.shape[-1]):
            # Extract the i-th layer of the model
            m_layer = M[:, :, :, :, i].to(self.device)

            # Compute the point spread function (PSF) for the current layer
            psf, center, _ = self.psf_layer(z)

            # Compute the FFT of the PSF kernel
            s_fft = self.fft_kernel(psf, center)

            # Compute the FFT of the model layer
            m_fft = torch.fft.fftshift(m_layer)
            m_fft = torch.fft.fft2(m_fft)
            m_fft = torch.fft.fftshift(m_fft)

            # Perform the convolution in the frequency domain
            b_fft = s_fft * m_fft
            b_fft = torch.fft.fftshift(b_fft)

            # Convert back to the spatial domain
            b_spatial = torch.real(torch.fft.ifft2(b_fft))

            # Accumulate the data from each layer
            data += b_spatial

            # Update depth
            z += dz

        return self.mudV * data

    def adjoint(self, data, height=0):
        """
        Perform the adjoint operation.

        Parameters:
            data (torch.Tensor): The observed magnetic data tensor.

        Returns:
            torch.Tensor: The adjoint result (model update).
        """
        dz = self.h[2]
        z = height + dz / 2  # Starting depth

        # Initialize the result tensor
        m_adj = torch.zeros(
            1, 1, self.dim[0], self.dim[1], self.dim[2], device=self.device
        )

        for i in range(self.dim[2]):
            # Compute the PSF for the current layer
            psf, center, _ = self.psf_layer(z)

            # Compute the FFT of the PSF kernel
            s_fft = self.fft_kernel(psf, center)

            # Compute the FFT of the input data
            data_fft = torch.fft.fft2(data)
            data_fft = torch.fft.fftshift(data_fft)

            # Perform the adjoint operation in the frequency domain
            b_fft = torch.conj(s_fft) * data_fft

            # Convert back to the spatial domain
            b_spatial = torch.fft.fftshift(b_fft)
            b_spatial = torch.real(torch.fft.ifft2(b_spatial))
            b_spatial = torch.fft.fftshift(b_spatial)

            # Store the result for the current layer
            m_adj[..., i] = b_spatial

            # Update depth
            z += dz

        return self.mudV * m_adj

    def psf_layer(self, z):
        """
        Compute the point spread function (PSF) for a layer at depth z.

        Parameters:
            z (float): The depth of the layer.

        Returns:
            psf (torch.Tensor): The computed PSF.
            center (list): Center indices for fftshift.
            rf (torch.Tensor): The radial factor (unused but computed for completeness).
        """
        # Unpack magnetic field directions
        I, A, I0, A0 = (
            self.dirs
        )  # Dip and declination angles for magnetization and geomagnetic field

        # Compute half-dimensions
        nx2, ny2 = self.dim[0] // 2, self.dim[1] // 2

        dx, dy = self.h[0], self.h[1]

        # Create coordinate grids
        x = dx * torch.arange(-nx2 + 1, nx2 + 1, device=self.device)
        y = dy * torch.arange(-ny2 + 1, ny2 + 1, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Center indices for fftshift
        center = [1 - nx2, 1 - ny2]

        # Compute the radial factor
        rf = (X**2 + Y**2 + z**2) ** 2.5

        # Compute components of the PSF
        cos_I = torch.cos(I)
        sin_I = torch.sin(I)
        cos_A = torch.cos(A)
        sin_A = torch.sin(A)
        cos_I0 = torch.cos(I0)
        sin_I0 = torch.sin(I0)
        cos_A0 = torch.cos(A0)
        sin_A0 = torch.sin(A0)

        PSFx = (
            (2 * X**2 - Y**2 - z**2) * cos_I * sin_A
            + 3 * X * Y * cos_I * cos_A
            + 3 * X * z * sin_I
        ) / rf

        PSFy = (
            3 * X * Y * cos_I * sin_A
            + (2 * Y**2 - X**2 - z**2) * cos_I * cos_A
            + 3 * Y * z * sin_I
        ) / rf

        PSFz = (
            3 * X * z * cos_I * sin_A
            + 3 * Y * z * cos_I * cos_A
            + (2 * z**2 - X**2 - Y**2) * sin_I
        ) / rf

        # Combine components to get the total PSF
        psf = PSFx * cos_I0 * cos_A0 + PSFy * cos_I0 * sin_A0 + PSFz * sin_I0

        return psf, center, rf


# %%
class TimeEmbKernel(nn.Module):
    def __init__(self, hid_dim=32, kernel_size=9, ker_hid_dim=128):
        """
        A time dependent 3D cubic kernel

        Parameters:
            hid_dim (int): Hidden dimension size, per kernel element.
            kernel_size (int): Size of the 3D kernel.
            ker_hid_dim (int): Hidden dimension size for the MLP layers.
        """
        super(TimeEmbKernel, self).__init__()
        self.kernel_size = kernel_size

        N = hid_dim * kernel_size**3
        self.b = nn.Parameter(1e-3 * torch.randn(N))
        self.C1 = nn.Linear(N, ker_hid_dim)
        self.C2 = nn.Linear(ker_hid_dim, N)

    def forward(self, t):
        k = self.kernel_size
        bt = F.silu(t * self.b)  # Si
        K = self.C2(F.silu(self.C1(bt)))
        K = K.view(1, -1, k, k, k)
        return K
    
class PushForward(nn.Module):
    def __init__(self):
        super(PushForward, self).__init__()

    def forward(self, T, UVW):
        device = T.device        
        D, H, W = T.shape[-3:]

        # Create normalized coordinate grid in [-1, 1]
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D, device=device),
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
        )
        grid = torch.stack((x, y, z), dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W, 3]

        transformation_grid = grid + UVW
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)

        return Th


# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n1, n2, n3 = 64, 64, 32
    adjoint_test = True

    if adjoint_test:
        dim = torch.tensor([n1, n2, n3])
        h = torch.tensor([100.0, 100.0, 100.0]).to(device)
        dirs = torch.tensor([np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]).to(device)
        forMod = Magnetics(dim, h, dirs, device=device)

        M = torch.rand(dim[0], dim[1], dim[2]).unsqueeze(0).unsqueeze(0).to(device)
        D = forMod(M)
        Q = torch.rand_like(D)
        W = forMod.adjoint(Q)

        print("Adjoint test:", torch.sum(M * W).item(), torch.sum(D * Q).item())
# %%
