"""
Implementation of TV regularization for the magnetic inverse problem.

References:
[1] Vogel, C. R. (2002). Total Variation Regularization. In Computational Methods for Inverse Problems (pp. 129–150).
Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898717570.ch8
[2] Ahamed, Shadab, et al. "Inversion of Magnetic Data using Learned Dictionaries and Scale Space." 
arXiv preprint arXiv:2502.05451 (2025).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

import pyvista as pv

import ssmaginv.config as config
from ssmaginv.magnetics import Magnetics
from ssmaginv.plot import plot_model_with_forward, plot_mixed

from torch.utils.data import DataLoader


def conj_gradient(
    A,
    b,
    x0=None,
    niter=20,
    tol=1e-2,
    verbose=True,
):
    """
    Solve Ax = b using the conjugate gradient method.

    Parameters:
        A (callable): A function that computes the operator A(x).
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): The initial guess. Defaults to None.
        niter (int, optional): Maximum number of iterations. Defaults to 20.
        tol (float, optional): Tolerance for the residual. Defaults to 1e-2.
        verbose (bool, optional): Whether to print progress. Defaults to True.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A(x)
    q = r.clone()
    for i in range(niter):
        Aq = A(q)
        alpha = (r * r).sum() / (q * Aq).sum()

        x = x + alpha * q
        r_new = r - alpha * Aq

        res_norm = r_new.norm() / b.norm()
        if verbose:
            print(f"iter = {i+1:3d}    res = {res_norm:.2e}")

        if res_norm < tol:
            break

        beta = (r_new**2).sum() / (r**2).sum()
        q = r_new + beta * q
        r = r_new.clone()
    return x


def gradx(x, h=(1.0, 1.0, 1.0)):
    assert len(h) == 3
    """ Return the forward gradient with loss of 1 element in each direction """
    gx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    gy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    gz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return gx / h[0], gy / h[1], gz / h[2]


def divergence_of_grad(gx, gy, gz, h=(1.0, 1.0, 1.0)):
    """
    Computes the divergence of a 3D vector field represented by gradients gx, gy, gz.
    Args:
        gx (torch.Tensor): Gradient along the x-direction (shape: [B, C, N1, N2, N3-1]).
        gy (torch.Tensor): Gradient along the y-direction (shape: [B, C, N1, N2-1, N3]).
        gz (torch.Tensor): Gradient along the z-direction (shape: [B, C, N1-1, N2, N3]).
        h (tuple): Grid spacings along (dx, dy, dz), defaults to (1, 1, 1).
    Returns:
        torch.Tensor: Divergence of the vector field (shape: [B, C, N1, N2, N3]).
    """
    # Unpack grid spacings
    dx, dy, dz = h

    # Get the dimensions for the output tensor
    b, c, n1, n2, n3 = (
        gz.shape[0],
        gz.shape[1],
        gz.shape[-3],
        gz.shape[-2],
        gx.shape[-1],
    )

    # Allocate output tensor for divergence
    divG = torch.zeros((b, c, n1, n2, n3), device=gx.device, dtype=gx.dtype)

    # z-component (gz divergence)
    divG[:, :, :, :, 0] -= gz[:, :, :, :, 0] / dz
    divG[:, :, :, :, -1] += gz[:, :, :, :, -1] / dz
    divG[:, :, :, :, 1:-1] += (gz[:, :, :, :, :-1] - gz[:, :, :, :, 1:]) / dz

    # y-component (gy divergence)
    divG[:, :, :, 0, :] -= gy[:, :, :, 0, :] / dy
    divG[:, :, :, -1, :] += gy[:, :, :, -1, :] / dy
    divG[:, :, :, 1:-1, :] += (gy[:, :, :, :-1, :] - gy[:, :, :, 1:, :]) / dy

    # x-component (gx divergence)
    divG[:, :, 0, :, :] -= gx[:, :, 0, :, :] / dx
    divG[:, :, -1, :, :] += gx[:, :, -1, :, :] / dx
    divG[:, :, 1:-1, :, :] += (gx[:, :, :-1, :, :] - gx[:, :, 1:, :, :]) / dx

    return divG


def cell_average(ux, uy, uz, h=(1.0, 1.0, 1.0)):
    """
    Input is X-1, Y-1, Z-1, from gradient, averageing loss of 1 element in each direction
    X-2, Y-2, Z-2, to select only cells that are not on the boundary (boundary is ignored)
    """
    uxa = 0.5 * (ux[:, :, 1:, 1:-1, 1:-1] + ux[:, :, :-1, 1:-1, 1:-1])
    uya = 0.5 * (uy[:, :, 1:-1, 1:, 1:-1] + uy[:, :, 1:-1, :-1, 1:-1])
    uza = 0.5 * (uz[:, :, 1:-1, 1:-1, 1:] + uz[:, :, 1:-1, 1:-1, :-1])
    return uxa, uya, uza


def cell_average_adjoint(vx, vy, vz, h=(1.0, 1.0, 1.0)):
    n1, n2, n3 = vz.shape[-3] + 2, vz.shape[-2] + 2, vx.shape[-1] + 2
    b = vx.shape[0]
    c = vx.shape[1]
    device, dtype = vx.device, vx.dtype

    x1 = torch.zeros(
        b, c, n1 - 1, n2, n3, device=device, dtype=dtype, requires_grad=True
    )
    x2 = torch.zeros(
        b, c, n1, n2 - 1, n3, device=device, dtype=dtype, requires_grad=True
    )
    x3 = torch.zeros(
        b, c, n1, n2, n3 - 1, device=device, dtype=dtype, requires_grad=True
    )

    bx, by, bz = cell_average(x1, x2, x3, h)

    dot1 = torch.sum(bx * vx)
    dot2 = torch.sum(by * vy)
    dot3 = torch.sum(bz * vz)

    ATvx = torch.autograd.grad(dot1, x1, create_graph=True)[0]
    ATvy = torch.autograd.grad(dot2, x2, create_graph=True)[0]
    ATvz = torch.autograd.grad(dot3, x3, create_graph=True)[0]

    return ATvx, ATvy, ATvz


def compute_total_variation(u, h=(1.0, 1.0, 1.0), eps=1e-2):
    # Compute the derivatives
    gx, gy, gz = gradx(u, h)
    gxs = gx**2
    gys = gy**2
    gzs = gz**2

    gxs, gys, gzs = cell_average(gxs, gys, gzs, h)
    graduSq = gxs + gys + gzs
    TV = torch.sqrt(graduSq + eps)
    V = h[0] * h[1] * h[2]
    TV = (V * TV.sum(dim=(1, 2, 3, 4))).mean()
    return TV


def Lf_v(u, fv, h=(1.0, 1.0, 1.0), eps=1e-2):
    """Lv operator from [1] Algorithm 8.2.3, takes fv as input"""
    # application of grad operator to fv
    gvx, gvy, gvz = gradx(fv, h)

    # Compute the inner matrix using TV
    gx, gy, gz = gradx(u, h)
    gxs = gx**2
    gys = gy**2
    gzs = gz**2

    gxs, gys, gzs = cell_average(gxs, gys, gzs, h)
    graduSq = gxs + gys + gzs
    TV = torch.sqrt(graduSq + eps)

    sigx, sigy, sigz = cell_average_adjoint(1 / TV, 1 / TV, 1 / TV)

    # Apply inner matrix to grad(fv)
    Jx, Jy, Jz = sigx * gvx, sigy * gvy, sigz * gvz

    # Apply the adjoint (divergence) operator to the result
    divJ = divergence_of_grad(Jx, Jy, Jz)

    return divJ


def general_adjoint(A, v, x_sample):
    """
    Take the adjoint of the forward operator A with respect to the input vector v.

    Parameters:
        A (callable): Forward operator.
        v (torch.Tensor): Input vector.
        x_sample (torch.Tensor): Sample input for dimensioning (dummy data).
    """

    x = torch.zeros_like(x_sample)
    x.requires_grad = True
    b = A(x)
    # Compute the dot product of the forward operator with the input vector
    h = torch.sum(b * v)
    # Compute the gradient of the dot product with respect to the input image
    adjoint = torch.autograd.grad(h, x, create_graph=True)[0]
    return adjoint


def solve_ls(x0, D, forMod, reg_func=gradx, alpha=1e-6, misfit_tol=1e-3):
    """
    Solution of the minimizer of |Ax - b|^2 + alpha * |Rx|^2

    Parameters:
        x0 (torch.Tensor): Initial guess for the model.
        D (torch.Tensor): Data vector (b) to be matched.
        forMod (callable): Forward operator function (A) to compute Ax, requires adjoint.
        reg_func (callable): Regularization function (R) to compute Rx.
        alpha (float): Regularization parameter.
        misfit_tol (float): Tolerance for the misfit condition.
    """

    # Define the forward operator with regularization
    def A(x, alpha=alpha):
        y1 = forMod(x)
        y1 = forMod.adjoint(y1)
        y2 = reg_func(x)
        y2 = general_adjoint(reg_func, y2, x_sample=x)
        return y1 + alpha * y2

    b = forMod.adjoint(D)

    # Check dimensions and functioning
    try:
        y = A(torch.randn_like(x0))
        print(f"Dimensions of A(xtrue): {y.shape}")
        print(f"Dimensions of b: {b.shape}")
        c = y - b
    except Exception as e:
        print(f"Error: Dimensions of A(xtrue) and b do not match.")
        raise e

    # Set the break condition as being within a tolerance misfit
    def break_condition(x):
        return 0.5 * (D - forMod(x)).norm() ** 2 < misfit_tol

    # Solve the inverse problem using the conjugate gradient method
    xinv = conj_gradient(
        A,
        b,
        x0,
        break_condition_func=break_condition,
        niter=160,
        tol=1e-6,
        alpha=1e-2,
        verbose=True,
    )

    # Verify that the gradient is zero at the solution
    xtest = xinv.clone().detach().requires_grad_(True)
    loss = 0.5 * (D - forMod(xtest)).norm() ** 2 + 0.5 * alpha * torch.sum(
        reg_func(xtest) ** 2
    )
    loss.backward()

    # Print the norm of the gradient
    gradient_norm = xtest.grad.norm().item()
    print(f"Gradient norm at xinv: {gradient_norm:.6e}")

    # Get misfit and regularization
    misfit = 0.5 * (D - forMod(xinv)).norm() ** 2
    reg = 0.5 * alpha * torch.sum(reg_func(xtest) ** 2)
    print(f"Misfit: {misfit:.6e}, Regularization: {reg:.6e}")

    # Optionally, set a tolerance and assert
    gradient_tolerance = 1e-4
    if gradient_norm < gradient_tolerance:
        print(
            f"Verification Passed: Gradient norm {gradient_norm:.2e} is below the tolerance {gradient_tolerance:.2e}."
        )
    else:
        print(
            f"Verification Failed: Gradient norm {gradient_norm:.2e} exceeds the tolerance {gradient_tolerance:.2e}."
        )

    return xinv


def set_z_weights(zdim, zscale, z0):
    """
    Set the weighting for the z-layer regularization penalty term.

    Parameters:
        zdim (int): Number of z layers in the model.
        zscale (float): Scaling factor for the z weights.
        z0 (float): Offset for the z weights.
    Returns:
    """
    # make a vector of the z values of the model usin gthe zscale
    Z_weights = torch.range(0, zdim - 1, dtype=torch.float32) * zscale
    Z_weights = 1 / (Z_weights + z0) ** (1.5)  # Li and Oldenbur 1996 weighting
    return Z_weights


def solve_tv(
    x0,
    dim,
    D,
    forMod,
    alpha=1e-2,
    Z_weights=None,
    beta=1e-2,
    n_iters=10,
    misfit_tol=1e-4,
    verbose=False,
):
    """
    Eq. 3 from [2].
    Solves for the TV regularized inverse problem using the conjugate gradient method.
    Includes an additional z-weighting regularization term to promote response far from the surface.


    Parameters:
        x0 (torch.Tensor): Initial guess for the model.
        dim (torch.Tensor): Dimensions of the model.
        D (torch.Tensor): Data vector (b) to be matched.
        forMod (callable): Forward operator function (A) to compute Ax, requires adjoint.
        alpha (float): Weighting parameter for the total varaition regularization term.
        Z_weights (torch.Tensor, optional): Weights for the z-direction regularization.
        beta (float): Weighting parameter for the z-weighting regularization.
        n_iters (int): Number of iterations for the conjugate gradient method.
        misfit_tol (float): Tolerance for the misfit condition.
        verbose (bool): Whether to print progress.
    """

    zdim = x0.shape[-1]
    device = x0.device

    if Z_weights is None:
        Z_weights = torch.zeros(zdim, device=device)
    else:
        Z_weights = Z_weights.to(device)

    z_reg = lambda x: beta * (Z_weights**2 * x)

    # Initial guess
    x = x0.clone().detach().to(device).requires_grad_(False)

    for i in range(n_iters):
        # Gradient
        gv = forMod.adjoint(forMod(x) - D) + alpha * Lf_v(x, x, h=dim) + z_reg(x)

        # Solve system (A^TA + alpha Lf_v) s = -gv with conj gradient
        def lh_operator(s, alpha=alpha):
            y1 = forMod(s)
            y1 = forMod.adjoint(y1)
            y2 = Lf_v(x, s, h=dim)
            y3 = z_reg(s)
            return y1 + alpha * y2 + y3

        s = conj_gradient(
            lh_operator, -gv, x0=torch.zeros_like(x), niter=40, tol=1e-4, verbose=False
        )
        x = x + s

        # Check misfit
        misfit = 0.5 * (D - forMod(x)).norm() ** 2
        if verbose:
            print(f"Misfit: {misfit:.6e}")
        if misfit < misfit_tol:
            print(f"Converged at iteration {i}")
            break

    return x
