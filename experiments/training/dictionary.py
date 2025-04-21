import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

from ssmaginv.magnetics import Magnetics
from ssmaginv.dataset.saved_dataset import get_train_set, get_valid_set
from ssmaginv.solvers.dictionary import ScaleSpaceRecovery, SingleDictRecovery
from ssmaginv.config import get_default_magnetics_config, RESULTS_DIR


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def build_loaders(bs=16):
    tr, va = get_train_set(), get_valid_set()
    if tr is None or va is None:
        raise RuntimeError("Run reproduce_ss_paper_data() first (datasets missing).")
    kwargs = dict(num_workers=0, pin_memory=True)
    return (
        DataLoader(tr, bs, shuffle=True, **kwargs),
        DataLoader(va, bs, shuffle=False, **kwargs),
    )


def batch_losses(forOp, x_pred, x_true, Ax, data):
    lossD = ((Ax - data) ** 2).mean() / (data**2).mean()
    lossX = ((x_pred - x_true) ** 2).mean() / (x_true**2).mean()
    return lossD, lossX, (lossD + lossX)


def log_step(csv_path, row):
    hdr = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(row.keys())
        w.writerow(row.values())


# ---------------------------------------------------------------------
# generic trainer for dictionary models
# ---------------------------------------------------------------------
def train_model(
    model_name: str,
    model_ctor,  # callable -> nn.Module
    num_epochs=200,
    batch_size=16,
    lr=1e-2,
    val_interval=1,
):

    # --- folders ------------------------------------------------------
    log_dir = os.path.join(RESULTS_DIR, "LOGS")
    weight_dir = os.path.join(RESULTS_DIR, "WEIGHTS")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, f"{model_name}.csv")
    best_ckpt = os.path.join(weight_dir, f"{model_name}.pt")

    # --- data ---------------------------------------------------------
    dl_train, dl_val = build_loaders(batch_size)

    # --- operators & model -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model {model_name} on {device}")
    cfg = get_default_magnetics_config()
    cfg["device"] = device
    forOp = Magnetics(**cfg).to(device)
    model = model_ctor(forOp=forOp, device=device).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-5)

    best_val = float("inf")
    va_D = va_X = va_T = 0.0
    # --- training loop -----------------------------------------------
    for epoch in range(1, num_epochs + 1):

        # ----------------------- train -------------------------------
        model.train()
        tr_D = tr_X = tr_T = 0.0  # Losses
        with tqdm(
            enumerate(dl_train),
            total=len(dl_train),
            desc=f"Epoch {epoch}/{num_epochs}",
            unit="batch",
        ) as progress_bar:
            for i, (x_true, data_true, data_noisy) in progress_bar:
                opt.zero_grad()
                # Prepare input and target data
                x_true = x_true.to(device)
                data_true = data_true.to(device)
                data_noisy = data_noisy.to(device)

                z_latent, x_pred, d_pred = model.forward(data_noisy)

                lossD, lossX, lossTot = batch_losses(
                    forOp, x_pred, x_true, d_pred, data_true
                )
                lossTot.backward()
                opt.step()

                tr_D += lossD.item()
                tr_X += lossX.item()
                tr_T += lossTot.item()
                
                progress_bar.set_postfix(
                    LossD=f'{lossD.item():.6f}', 
                    LossX=f'{lossX.item():.6f}', 
                    Loss=f'{lossTot.item():.6f}')

        n_tr = len(dl_train)
        tr_D /= n_tr
        tr_X /= n_tr
        tr_T /= n_tr
        print(f"[train]  Ltot={tr_T:.3e}  LD={tr_D:.3e}  LX={tr_X:.3e}")

        if epoch % val_interval == 0:  # <‑‑ only every j‑th epoch
            model.eval()
            va_D = va_X = va_T = 0.0
            with torch.no_grad():
                for x_true, _, data_noisy in dl_val:
                    x_true, data_noisy = x_true.to(device), data_noisy.to(device)
                    _, x_pred, d_pred = model(data_noisy)
                    lD, lX, lT = batch_losses(forOp, x_pred, x_true, d_pred, data_noisy)
                    va_D += lD.item()
                    va_X += lX.item()
                    va_T += lT.item()

            n_val = len(dl_val)
            va_D /= n_val
            va_X /= n_val
            va_T /= n_val
            print(f"[valid]  Ltot={va_T:.3e}  LD={va_D:.3e}  LX={va_X:.3e}")

        # ---------- log row ----------
        log_step(
            log_csv,
            dict(
                epoch=epoch,
                tr_data=tr_D,
                tr_model=tr_X,
                tr_total=tr_T,
                va_data=va_D,
                va_model=va_X,
                va_total=va_T,
            ),
        )

        # ---------- checkpoint ----------
        if va_T < best_val:
            best_val = va_T
            torch.save(model.state_dict(), best_ckpt)
            print(f"New best val loss: {best_val}, saved to {best_ckpt}")

        sched.step()


# ---------------------------------------------------------------------
# convenience wrappers
# ---------------------------------------------------------------------
def train_single_dict(bs=16):
    train_model("SingleDict", SingleDictRecovery, batch_size=bs)


def train_scale_space(bs=16):
    train_model("ScaleSpace", ScaleSpaceRecovery, batch_size=bs)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["single", "scale"], required=True)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    if args.model == "single":
        train_single_dict(bs=args.bs)
    else:
        train_scale_space(bs=args.bs)
