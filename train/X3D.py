#!/usr/bin/env python3

import pathlib
import logging
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from datasets import MultiNpyEdf
from utils import save_model_weights, save_loss, downsample, write_dict_to_csv
from modules import X3D_head, Enhanced_X3D_head
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


def load(
    dataset: Dataset,
    dataset_path: pathlib.Path,
    schema_json: pathlib.Path,
    model: str,
    b_size=5,
    shuffle=True,
    n_workers=2,
    downsample_classes=False,
    downsample_seed=0,
) -> DataLoader:
    """
    Load the dataset and optionally downsample it.
    """
    if model == "OF_X3D":
        npy_files = dataset_path.rglob("0flow_*x3d*.npy")
    else:
        npy_files = dataset_path.rglob("0rgb_*x3d*.npy")
    edf_files = dataset_path.rglob("*.edf")
    data = dataset(npy_files, edf_files, schema_json, window_size=75)

    # Downsample the dataset if required
    if downsample_classes:
        logging.info("Downsampling the dataset for balanced classes.")
        data = downsample(data, seed=downsample_seed, verbose=True)

    return DataLoader(
        data,
        batch_size=b_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=True,
    )


def calculate_pos_weight(dataset: Dataset) -> torch.Tensor:
    """
    Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance.
    """
    labels = [dataset[idx][1] for idx in range(len(dataset))]
    class_counts = np.bincount(labels)
    pos_weight = class_counts[0] / \
        (class_counts[1] + 1e-6)  # Adjust for imbalance
    return torch.tensor(pos_weight, dtype=torch.float)


def init(args) -> tuple[nn.Module, DataLoader, torch.Tensor]:
    """
    Initialize the model, data loader, and pos_weight.
    """
    arch = {"RGB_X3D": X3D_head, "OF_X3D": X3D_head}
    model = arch[args.model](num_classes=args.target,
                             dropout_prob=args.dropout)
    dataloader = load(
        MultiNpyEdf,
        args.data_path,
        args.schema_path,
        args.model,
        b_size=args.batch_size,
        shuffle=args.shuffle,
        downsample_classes=args.downsample,
        downsample_seed=args.downsample_seed,
    )

    # Calculate pos_weight
    pos_weight = calculate_pos_weight(dataloader.dataset)

    return model, dataloader, pos_weight


def train(
    args,
    device: torch.device,
    n_epochs: int = 1,
) -> nn.Module:
    """
    Train the model with BCEWithLogitsLoss.
    """
    model, dataloader, pos_weight = init(args)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # pos_weight=pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{n_epochs}.")
        epoch_loss = 0  # Track cumulative loss for the epoch
        num_batches = len(dataloader)

        for batch_number, data in enumerate(dataloader):
            try:
                X, y = data
                optimizer.zero_grad()
                # BCE requires float targets
                X, y = X.to(device), y.to(device).float()
                y_pred = model(X)
                # Squeeze logits for BCE compatibility
                loss = loss_fn(y_pred.squeeze(), y)
                loss.backward()
                optimizer.step()

                # Update cumulative epoch loss
                epoch_loss += loss.item()

                # Save batch loss
                save_loss(
                    loss.item(),
                    args.path_to_model_save /
                    f"loss_epoch{epoch+1}_batch{batch_number+1}.txt",
                )
            except Exception as e:
                logging.error(f"Error in batch {batch_number + 1}: {e}")

        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        logging.info(
            f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}"
        )

        # Step the learning rate scheduler
        scheduler.step()
        # optimizer.step()
    return model


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute evaluation metrics for binary classification.
    Includes accuracy, precision, recall, F1-score, ROC-AUC, and weighted metrics.
    """
    metrics = {}

    # Flatten arrays if needed
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    # Metrics calculations
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)

    # Weighted metrics
    metrics['weighted_precision'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_recall'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_f1_score'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0)

    # F-beta score for prioritizing recall or precision
    metrics['fbeta_0.5'] = fbeta_score(
        y_true, y_pred, beta=0.5, zero_division=0)
    metrics['fbeta_2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    return metrics


def evaluate(args, model, device) -> None:
    """
    Evaluate the model on the test set and compute metrics.
    """
    logging.info("Starting evaluation...")
    dataloader = load(
        MultiNpyEdf,
        args.testset,
        args.schema_path,
        args.model,
        b_size=args.batch_size,
        shuffle=False,  # No shuffling for evaluation
    )

    y_predictions, y_true = [], []
    model.eval()

    with torch.no_grad():
        for batch_number, data in enumerate(dataloader):
            try:
                X, y = data
                X, y = X.to(device), y.to(device)
                y_true.append(y.cpu().numpy())
                # Apply sigmoid for probabilities
                y_pred = torch.sigmoid(model(X))
                # Threshold at 0.5
                y_predictions.append((y_pred > 0.5).cpu().numpy())
            except Exception as e:
                logging.error(f"Error in batch {batch_number + 1}: {e}")

    y_true = np.concatenate(y_true, axis=0)
    y_predictions = np.concatenate(y_predictions, axis=0)

    metrics = compute_metrics(y_true, y_predictions)
    write_dict_to_csv(metrics, args.path_to_model_save /
                      "evaluation_metrics.csv", write_headers=True)

    logging.info(metrics)
    logging.info("Evaluation metrics saved to evaluation_metrics.csv")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("schema_path", type=pathlib.Path)
    parser.add_argument("--path_to_model_save", type=pathlib.Path,
                        default=pathlib.Path("./model_weights"))
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    # Binary classification
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--model", type=str, default="RGB_X3D")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the dataset during training.")
    parser.add_argument("--testset", type=pathlib.Path, default=None)
    parser.add_argument("--downsample", action="store_true",
                        help="Enable downsampling for class balance.")
    parser.add_argument("--downsample_seed", type=int,
                        default=0, help="Random seed for downsampling.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(args, device=device, n_epochs=args.epochs)
    save_model_weights(model, args.path_to_model_save /
                       f"{args.model}_lr{args.learning_rate}_epochs{args.epochs}.pth")

    if args.testset:
        evaluate(args, model, device)


if __name__ == "__main__":
    main()
