"""Resolve compute device for hybrid ranker training."""

from __future__ import annotations

import torch


def resolve_device(device_setting: str) -> torch.device:
    """
    Pick the torch device used for hybrid ranker training.

    Do this by:
    1. Using CUDA when device_setting is auto and a GPU is available.
    2. Otherwise using the explicit device string from settings.

    ============================ Arguments ============================
    device_setting: One of auto, cpu, or cuda.

    ============================ Returns ============================
    The resolved torch.device.
    """
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)
