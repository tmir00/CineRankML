"""Plot CF training curves for artifact export."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence


def save_training_curve(output_path: Path, *, train_losses: Sequence[float], validation_rmses: Sequence[float], 
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
    # Create a list of epochs.
    epochs = list(range(1, len(train_losses) + 1))
    # Create a figure and axis.
    fig, axis = plt.subplots(figsize=(8, 5))
    # Plot the train loss on the primary y-axis.
    axis.plot(epochs, train_losses, label="train_loss", color="tab:blue")
    # Set the x-label to "epoch".
    axis.set_xlabel("epoch")
    # Set the y-label to "train_loss".
    axis.set_ylabel("train_loss", color="tab:blue")
    # Set the tick parameters for the primary y-axis.
    axis.tick_params(axis="y", labelcolor="tab:blue")

    # Create a secondary y-axis for validation metrics.
    # Plot the validation RMSE on the secondary y-axis.
    axis2 = axis.twinx()
    # Plot the validation MAE on the secondary y-axis.
    axis2.plot(epochs, validation_rmses, label="validation_rmse", color="tab:orange")
    axis2.plot(epochs, validation_maes, label="validation_mae", color="tab:green")
    # Set the y-label to "validation metrics".
    axis2.set_ylabel("validation metrics")


    lines = axis.get_lines() + axis2.get_lines()
    labels = [line.get_label() for line in lines]
    # Create a legend for the lines.
    axis.legend(lines, labels, loc="upper right")

    # Save the figure to the output path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
