#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path
import csv


def write_dict_to_csv(data_dict, file_path, write_headers=True):
    """
    Writes a dictionary to a CSV file. If the file does not exist, it creates it and writes headers based on the dictionary keys.

    Args:
    data_dict (dict): The dictionary to write to the CSV file. All values should be scalar (no lists or nested dictionaries).
    file_path (str): The path to the CSV file where the data will be written.
    write_headers (bool): If True, writes the headers (dictionary keys); useful for the first time writing to a file.
    """
    # Open the file in append mode, create it if it does not exist ('a+')
    with open(file_path, 'a+', newline='') as file:
        # Create a CSV DictWriter object
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())

        # Move the reader to the end of the file and check if it's empty
        file.seek(0, 2)
        if file.tell() == 0:
            write_headers = True

        # Write headers if required
        if write_headers:
            writer.writeheader()

        # Write the dictionary as one row in the CSV file
        writer.writerow(data_dict)


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


def save_loss(loss_value, file_path):
    """
    Appends a given loss value to a specified file.

    Args:
    loss_value (float): The loss value to be saved.
    file_path (str): The path to the file where the loss values are stored.
    """
    with open(file_path, 'a') as file:
        file.write(f"{loss_value}\n")
