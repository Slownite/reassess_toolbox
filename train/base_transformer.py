#!/usr/bin/env python3

import pathlib
import logging
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import MultiNpyEdfSequence
from utils import save_model_weights, save_loss, downsample, write_dict_to_csv
from modules import ProjectedTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    fbeta_score
)
import numpy as np

logging.basicConfig(level=logging.INFO)


def collate_fn(batch):
    sequences, labels = zip(*batch)  # Separate sequences and labels
    # Pad sequences so they all have the same length
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)  # Stack labels into a single tensor
    return padded_sequences, labels


def load_dataset(
    dataset_path: pathlib.Path,
    schema_json: pathlib.Path,
    sequence_length: int = 10,
    downsample_classes=False,
    downsample_seed=0,
) -> DataLoader:
    """
    Load the dataset using MultiNpyEdfSequence.
    """
    npy_files = dataset_path.rglob("0rgb_*x3d*.npy")
    edf_files = dataset_path.rglob("*.edf")

    dataset = MultiNpyEdfSequence(
        npy_files, edf_files, schema_json, sequence_length=sequence_length)
    if downsample_classes:
        logging.info("Downsampling the dataset for balanced classes.")
        dataset = downsample(dataset, seed=downsample_seed, verbose=True)
    return dataset


def split_dataset(dataset: Dataset, test_split: float, seed: int = 0):
    """
    Split the dataset into training and testing sets.
    """
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(
        dataset, [train_size, test_size], generator=torch.manual_seed(seed))
    return train_data, test_data


def init(cfg: DictConfig) -> tuple[nn.Module, DataLoader, DataLoader]:
    """
    Initialize the model and data loaders.
    """
    model = ProjectedTransformer(
        input_dim=cfg.transformer.input_dim,
        d_model=cfg.transformer.d_model,
        nhead=cfg.transformer.nhead,
        num_layers=cfg.transformer.num_layers,
        num_classes=cfg.transformer.num_classes
    )
    dataset = load_dataset(
        pathlib.Path(cfg.data_path),
        pathlib.Path(cfg.schema_path),
        sequence_length=cfg.sequence_length,
        downsample_classes=cfg.downsample,
        downsample_seed=cfg.seed,
    )

    train_data, test_data = split_dataset(
        dataset, cfg.test_split, seed=cfg.seed)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                              shuffle=cfg.shuffle, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    return model, train_loader, test_loader


def train(
    cfg: DictConfig,
    device: torch.device,
    n_epochs: int = 1,
) -> nn.Module:
    """
    Train the model.
    """
    model, train_loader, _ = init(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(
        optimizer, step_size=cfg.stepLR.step_size, gamma=cfg.stepLR.gamma)
    loss_fn = nn.BCEWithLogitsLoss()
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{n_epochs}.")
        epoch_loss = 0
        num_batches = len(train_loader)

        for batch_number, sequences in enumerate(train_loader):
            try:
                X = sequences[0]
                y = sequences[1]
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device).float()
                if torch.isnan(X).any() or torch.isinf(X).any():
                    logging.error(
                        f"NaN or Inf detected in input X at epoch {epoch}, batch {batch_number}")
                    continue
                y_pred = model(X)
                loss = loss_fn(y_pred.squeeze(), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                save_loss(loss.item(), pathlib.Path(cfg.path_to_model_save) /
                          f"loss_epoch{epoch+1}_batch{batch_number+1}.txt")
            except Exception as e:
                logging.error(f"Error in batch {batch_number + 1}: {e}")

        avg_epoch_loss = epoch_loss / num_batches
        logging.info(
            f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")
        scheduler.step()
    return model


@hydra.main(config_path="../conf", config_name="transformer.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(cfg, device=device, n_epochs=cfg.epochs)
    save_model_weights(model, pathlib.Path(cfg.path_to_model_save) /
                       f"{config_name}.pth")


if __name__ == "__main__":
    main()
