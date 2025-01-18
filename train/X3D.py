#!/usr/bin/env python3

import pathlib
import logging
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset, Subset
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from datasets import MultiNpyEdf
from utils import save_model_weights, save_loss, downsample, write_dict_to_csv
from modules import X3D_head, Enhanced_X3D_head
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
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

    Args:
        dataset (Dataset): Dataset class to load data.
        dataset_path (pathlib.Path): Path to the dataset.
        schema_json (pathlib.Path): Path to the schema JSON.
        model (str): Model type (e.g., RGB_X3D or OF_X3D).
        b_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        n_workers (int): Number of workers for DataLoader.
        downsample_classes (bool): Whether to downsample the dataset.
        downsample_seed (int): Seed for reproducibility.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    if model == "OF_X3D":
        npy_files = dataset_path.rglob("0flow_*x3d*.npy")
    else:
        npy_files = dataset_path.rglob("0rgb_*x3d*.npy")
    edf_files = dataset_path.rglob("*.edf")
    data = dataset(npy_files, edf_files, schema_json)

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


def calculate_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Calculate class weights for weighted cross-entropy.

    Args:
        dataset (Dataset): Dataset to calculate class weights from.

    Returns:
        torch.Tensor: Class weights for weighted cross-entropy.
    """
    labels = [dataset[idx][1] for idx in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights /= class_weights.sum()  # Normalize weights
    return torch.tensor(class_weights, dtype=torch.float)


def init(args) -> tuple[nn.Module, DataLoader, torch.Tensor]:
    """
    Initialize the model, data loader, and class weights.
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

    # Calculate class weights
    class_weights = calculate_class_weights(dataloader.dataset)

    return model, dataloader, class_weights


def train(
    args,
    device: torch.device,
    n_epochs: int = 1,
) -> nn.Module:
    """
    Train the model with the provided arguments.
    Logs loss at the end of each epoch.
    """
    model, dataloader, class_weights = init(args)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
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

    return model


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("schema_path", type=pathlib.Path)
    parser.add_argument("--path_to_model_save", type=pathlib.Path,
                        default=pathlib.Path("./model_weights"))
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--model", type=str, default="RGB_X3D")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--shuffle", type=bool, default=True)
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


if __name__ == "__main__":
    main()
