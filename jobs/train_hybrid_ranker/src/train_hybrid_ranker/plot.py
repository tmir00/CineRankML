"""Plot hybrid ranker training curves for artifact export."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence


def save_training_curve(output_path: Path, *, train_losses: Sequence[float], validation_rmses: Sequence[float], \
                            validation_maes: Sequence[float]) -> None:
    """
    Save a PNG plot of train loss and validation metrics versus epoch.

    Do this by:
    1. Plotting train loss on the primary y-axis.
    2. Plotting validation RMSE and MAE on shared epoch x-axis.
    3. Writing the figure to output_path.

    ============================ Arguments ============================
    output_path: Destination PNG path inside the artifact bundle.
    train_losses: Average training loss per epoch.
    validation_rmses: Validation RMSE per epoch.
    validation_maes: Validation MAE per epoch.
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, label="train_loss", color="tab:blue")
    axis.set_xlabel("epoch")
    axis.set_ylabel("train_loss", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")

    axis2 = axis.twinx()
    axis2.plot(epochs, validation_rmses, label="validation_rmse", color="tab:orange")
    axis2.plot(epochs, validation_maes, label="validation_mae", color="tab:green")
    axis2.set_ylabel("validation metrics")

    lines = axis.get_lines() + axis2.get_lines()
    labels = [line.get_label() for line in lines]
    axis.legend(lines, labels, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
