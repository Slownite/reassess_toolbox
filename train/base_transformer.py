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
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)  # Stack labels into a single tensor
    return padded_sequences, labels


def load_dataset(dataset_path: pathlib.Path, schema_json: pathlib.Path, sequence_length: int = 10, downsample_classes=False, downsample_seed=0) -> Dataset:
    npy_files = dataset_path.rglob("0rgb_*x3d*.npy")
    edf_files = dataset_path.rglob("*.edf")
    dataset = MultiNpyEdfSequence(
        npy_files, edf_files, schema_json, sequence_length=sequence_length)
    if downsample_classes:
        logging.info("Downsampling the dataset for balanced classes.")
        dataset = downsample(dataset, seed=downsample_seed, verbose=True)
    return dataset


def split_dataset(dataset: Dataset, test_split: float, seed: int = 0):
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(
        dataset, [train_size, test_size], generator=torch.manual_seed(seed))
    return train_data, test_data


def init(cfg: DictConfig) -> tuple[nn.Module, DataLoader, DataLoader]:
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


def compute_metrics(y_true, y_pred) -> dict:
    metrics = {}
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_pred),
        'Weighted Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'F-beta (0.5)': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'F-beta (2)': fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    })
    return metrics


def evaluate(cfg: DictConfig, config_name, model, test_loader, device):
    logging.info("Starting evaluation...")
    y_predictions, y_true = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_true.append(y.cpu().numpy())
            y_pred = torch.sigmoid(model(X))
            y_predictions.append((y_pred > 0.5).cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_predictions = np.concatenate(y_predictions, axis=0)
    metrics = compute_metrics(y_true, y_predictions)
    write_dict_to_csv(metrics, pathlib.Path(
        cfg.path_to_model_save) / f"{config_name}_evaluation_metrics.csv", write_headers=True)
    logging.info("Evaluation Results:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")
    logging.info("Evaluation metrics saved to evaluation_metrics.csv")


def train(cfg: DictConfig, config_name: str, device: torch.device, n_epochs: int = 1) -> nn.Module:
    model, train_loader, test_loader = init(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(
        optimizer, step_size=cfg.stepLR.step_size, gamma=cfg.stepLR.gamma)
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        num_batches = len(train_loader)
        logging.info(f"Starting epoch {epoch + 1}/{n_epochs}.")
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        logging.info(
            f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        scheduler.step()
    evaluate(cfg, config_name, model, test_loader, device)
    return model


@hydra.main(config_path="../conf", config_name="transformer.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(cfg, config_name, device=device, n_epochs=cfg.epochs)
    save_model_weights(model, pathlib.Path(cfg.path_to_model_save) /
                       f"{config_name}.pth")


if __name__ == "__main__":
    main()
