import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssmaginv.config import METRICS_DIR

N_SAMPLES = 100
CUTOFF = 1.2

# Style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
palette = sns.color_palette("deep")
var_color = palette[1]
ml_color = palette[0]


def load_losses(method_name):
    path = os.path.join(METRICS_DIR, method_name, "metrics.csv")
    df = pd.read_csv(path)
    return df["ModelLoss"].to_numpy(), df["DataLoss"].to_numpy()


def compare_methods(name_a, name_b, label_a, label_b, save_as):
    # Load data
    x_a, _ = load_losses(name_a)
    x_b, _ = load_losses(name_b)

    # Sanity check
    assert len(x_a) == len(x_b), "Mismatch in sample count between methods."

    # Random subset for visualization
    np.random.seed(0)
    sample_indices = np.random.choice(len(x_a), N_SAMPLES, replace=False)
    x_a = x_a[sample_indices]
    x_b = x_b[sample_indices]

    # Sort by Method A (typically VAR or SharedDict)
    sorted_idx = np.argsort(x_a)
    x_a_sorted = x_a[sorted_idx]
    x_b_sorted = x_b[sorted_idx]

    # Clip outliers
    x_a_clipped = np.minimum(x_a_sorted, CUTOFF)
    x_b_clipped = np.minimum(x_b_sorted, CUTOFF)

    # Detect outliers (for optional scatter markers)
    outliers_a = x_a_sorted > CUTOFF
    outliers_b = x_b_sorted > CUTOFF

    samples = np.arange(N_SAMPLES)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Lines between paired samples
    for i in range(N_SAMPLES):
        line_color = ml_color if x_b_sorted[i] < x_a_sorted[i] else var_color
        ax.plot([samples[i], samples[i]],
                [x_b_clipped[i], x_a_clipped[i]],
                color=line_color, alpha=0.6, linewidth=1.5)

    # Scatter points
    ax.scatter(samples[~outliers_b], x_b_clipped[~outliers_b],
               color=ml_color, label=label_b, s=10)
    ax.scatter(samples[~outliers_a], x_a_clipped[~outliers_a],
               color=var_color, label=label_a, s=10)

    # Labels and legend
    ax.set_xlabel("Samples (Sorted by " + label_a + " Loss)")
    ax.set_ylabel("Loss $\\|\\hat{m} - m\\|^2$")
    ax.set_ylim([0, CUTOFF + 0.2])
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_as, dpi=300, format="png", bbox_inches="tight")
    print(f"[✓] Saved figure to {save_as}")
    plt.show()


if __name__ == "__main__":
    # Comparison 1: Variational vs UnrolledDict
    compare_methods(
        name_a="Variational",
        name_b="UnrolledDict",
        label_a="Variational",
        label_b="Unrolled $\Psi$ Method",
        save_as="model_loss_comparison_var_vs_unrolled.png",
    )

    # Comparison 2: Shared (SingleDict) vs UnrolledDict
    compare_methods(
        name_a="SingleDict",
        name_b="UnrolledDict",
        label_a="Shared $\Psi$ Method",
        label_b="Unrolled $\Psi$ Method",
        save_as="model_loss_comparison_shared_vs_unrolled.png",
    )