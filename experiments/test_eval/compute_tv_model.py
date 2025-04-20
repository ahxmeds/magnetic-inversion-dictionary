import torch
from ssmaginv.magnetics import Magnetics
import ssmaginv.solvers.variational as l2
import ssmaginv.config as config

def compute_tv_model(x_true, data_true, alpha=1e-4, beta=1e3, z0=10):
    """
    Compute the total variation model for a given true model and data.
    
    Parameters::
        x_true (torch.Tensor): The true mag sus model tensor.
        data_true (torch.Tensor): The true mag forward data tensor.
        alpha (float): Regularization parameter for the total variation.
        beta (float): Regularization parameter for the z-regularization.
        z0 (float): Offset parameter for the z-weighting.
    """
    
    model_cfg = config.get_default_magnetics_config(device=x_true.device)  
    model_cfg["device"] = x_true.device  # Ensure the magnetics model is on the same device as x_true
    forMod = Magnetics(**model_cfg)  # Forward model operator
    D = data_true

    # Initial guess
    x0 = torch.zeros_like(x_true)

    # Create a z-weighting tensor
    dim = model_cfg["dim"]
    zdim = model_cfg["dim"]
    zscale = model_cfg["h"][-1]
    Z_weights = l2.set_z_weights(zdim=zdim, zscale=zscale, z0=z0)
    xinv = l2.solve_tv(
        x0,
        dim,
        D,
        forMod,
        alpha=alpha,
        Z_weights=Z_weights,
        beta=beta,
        n_iters=20,
        verbose=True,
    )

    # Prediced forward model
    pred = forMod(xinv)  # predicted 2d data
    lossd = torch.mean((D - pred) ** 2) / torch.mean((D) ** 2)
    lossx = torch.mean((xinv - x_true) ** 2) / torch.mean((x_true) ** 2)

    return xinv, pred, lossx, lossd