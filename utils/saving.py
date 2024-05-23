#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path


def save_model_weights(model, file_path):
    """
    Save the weights of a PyTorch model to a specified file path using pathlib.

    Parameters:
    model (torch.nn.Module): The PyTorch model whose weights are to be saved.
    file_path (str or Path): The file path where the model weights will be saved.
    """
    # Ensure file_path is a Path object
    file_path = Path(file_path)

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model's state_dict to the specified file path
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")
