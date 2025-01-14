#!/usr/bin/env python3

from argparse import ArgumentParser
import pathlib
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR
import torch
from datasets import MultiNpyEdf
from utils import save_model_weights, save_loss, downsample, write_dict_to_csv
from modules import RGB_I3D_head, OF_I3D_head
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np


def load(
    dataset: Dataset,
    dataset_path: pathlib.Path,
    schema_json: pathlib.Path,
    b_size=5,
    shuffle=True,
    n_workers=2,
) -> Dataset:
    npy_files = dataset_path.rglob("*.npy")
    edf_files = dataset_path.rglob("*.edf")
    data = dataset(npy_files, edf_files, schema_json)
    dataloader = DataLoader(
        data,
        batch_size=b_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dataloader


def init(args) -> tuple[nn.Module, DataLoader, nn.Module]:
    arch = {"RGB_I3D": RGB_I3D_head, "OF_I3D": OF_I3D_head}
    model = arch[args.model](args.target)
    dataloader = load(
        MultiNpyEdf,
        args.data_path,
        args.schema_path,
        policy=args.policy,
        b_size=args.batch_size,
        shuffle=args.shuffle,
        n_workers=args.workers,
    )

    return (model, dataloader)


def train(
    args,
    device: torch.device,
    n_epochs: int = 1,
) -> nn.Module:
    model, dataloader = init(args)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0015, 662.7814])).to(device)
    model = model.to(device)
    model.train()
    for i in range(n_epochs):
        for batch_number, data in enumerate(dataloader):
            print(f"start batch {batch_number}")
            X, y = data
            optimizer.zero_grad()  # Use optimizer.zero_grad() instead of scheduler.zero_grad()
            X_rgb = X[0]
            X_f = X[1]
            X_rgb = X_rgb.to(device).unsqueeze(2).unsqueeze(2)
            X_f = X_f.to(device).unsqueeze(2).unsqueeze(2)
            y = y.to(device)
            y_pred = model(X_rgb, X_f)
            loss = loss_fn(y_pred, y)
            print("loss:", loss.item())
            loss.backward()
            optimizer.step()  # Perform an optimizer step for each batch
            save_loss(loss.item(), args.path_to_model_save /
                      f"loss_{model}_{args.learning_rate}_{args.epochs}.txt".replace("\n", ""))
            print(f"batch {batch_number} done")
        scheduler.step()  # Step the scheduler once per epoch
    return model


def compute_metrics(args, model, y_pred, y_true):
    # Compute metric
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        specificity = tn / (tn + fp)

        # Compute ROC AUC
        if len(np.unique(y_true)) == 2:  # Check if it's a binary classification
            roc_auc = roc_auc_score(y_true, y_pred)
        else:
            roc_auc = "Not Applicable for multi-class"

        results = {
            "Model": f"{model}_{args.learning_rate}_{args.epochs}",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score": f2,
            "Specificity": specificity,
            "ROC AUC": roc_auc
        }
        write_dict_to_csv(results, 'results.csv', write_headers=True)
        print('result saved in the file results.csv')
    except UndefinedMetricWarning as umw:
        print(f"Warning: {umw}")


def evaluate(args, model, device) -> None:
    dataloader = load(
        MultiNpyEdf,
        args.testset,
        args.schema_path,
        policy=args.policy,
        b_size=args.batch_size,
        shuffle=args.shuffle,
        n_workers=args.workers,
        downsampling=False
    )
    y_predictions = []
    y_true = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch_number, data in enumerate(dataloader):
            print(f"start batch {batch_number}")
            try:
                X, y = data
                # Ensure tensors are properly structured and moved to device
                X_rgb = X[0].to(device).unsqueeze(2).unsqueeze(2)
                X_f = X[1].to(device).unsqueeze(2).unsqueeze(2)
                y = y.to(device)

                # Store true labels (move to CPU and convert to numpy)
                y_true.append(y.cpu().numpy())

                # Make predictions
                y_pred = model(X_rgb, X_f)
                y_pred = y_pred.argmax(dim=1).detach().cpu().numpy()
                y_predictions.append(y_pred)

                print(f"end batch {batch_number}")
            except Exception as e:
                print(f"Error in batch {batch_number}: {e}")
                continue  # Skip the problematic batch

    # Concatenate results for metrics
    y_true = np.concatenate(y_true, axis=0)
    y_predictions = np.concatenate(y_predictions, axis=0)

    # Compute metrics
    compute_metrics(args, model, y_predictions, y_true)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("schema_path", type=pathlib.Path)
    parser.add_argument(
        "-p",
        "--path_to_model_save",
        type=pathlib.Path,
        default=pathlib.Path("./model_weights"),
    )
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-t", "--target", type=int, default=2)
    parser.add_argument("-m", "--model", type=str, default="RGB_I3D")
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-wk", "--workers", type=int, default=0)
    parser.add_argument("-s", "--shuffle", type=bool, default=True)
    parser.add_argument("--testset", type=pathlib.Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(
        args, device=device, n_epochs=args.epochs
    )
    save_model_weights(model, args.path_to_model_save /
                       f"{model}_{args.learning_rate}_{args.epochs}.pth")
    if args.testset:
        evaluate(args, model, device)


if __name__ == "__main__":
    main()
