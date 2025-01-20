#!/usr/bin/env python3

from argparse import ArgumentParser
import pathlib
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from datasets import MultiNpyEdf
from utils import save_model_weights, save_loss, downsample, write_dict_to_csv
from modules import RGB_I3D_head, OF_I3D_head
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np


def load(
    dataset: Dataset,
    dataset_path: pathlib.Path,
    schema_json: pathlib.Path,
    model: str,
    b_size=5,
    shuffle=True,
    n_workers=2,
) -> Dataset:
    if model == "OF_I3D":
        npy_files = dataset_path.rglob("flow_*.npy")
    else:
        npy_files = dataset_path.rglob("rgb_*.npy")
    edf_files = dataset_path.rglob("*.edf")
    data = dataset(npy_files, edf_files, schema_json)
    return data


def split_data(dataset, test_percentage):
    total_size = len(dataset)
    test_size = int(total_size * test_percentage / 100)
    train_size = total_size - test_size
    return random_split(dataset, [train_size, test_size])


def get_dataloaders(dataset, batch_size, workers, shuffle):
    train_data, test_data = dataset
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=shuffle, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def init(args) -> tuple[nn.Module, DataLoader, nn.Module]:
    arch = {"RGB_I3D": RGB_I3D_head, "OF_I3D": OF_I3D_head}
    model = arch[args.model](args.target)
    full_dataset = load(
        MultiNpyEdf,
        args.data_path,
        args.schema_path,
        args.model,
        b_size=args.batch_size,
        shuffle=args.shuffle,
        n_workers=args.workers,
    )

    # Downsample dataset if specified
    if args.downsample:
        full_dataset = downsample(
            full_dataset, target_size=args.downsample_size, verbose=True)

    train_dataset, test_dataset = split_data(
        full_dataset, args.test_percentage)
    train_loader, test_loader = get_dataloaders(
        (train_dataset, test_dataset), args.batch_size, args.workers, args.shuffle)
    return model, train_loader, test_loader


def train(
    args,
    device: torch.device,
    n_epochs: int = 1,
) -> nn.Module:
    model, train_loader, _ = init(args)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0015, 662.7814])).to(device)
    model = model.to(device)
    model.train()
    for i in range(n_epochs):
        for batch_number, data in enumerate(train_loader):
            print(f"start batch {batch_number}")
            X, y = data
            optimizer.zero_grad()
            X_rgb = X
            X_f = X
            X_rgb = X_rgb.to(device)
            X_f = X_f.to(device)
            y = y.to(device)
            y_pred = model(X_rgb, X_f)
            loss = loss_fn(y_pred, y)
            print("loss:", loss.item())
            loss.backward()
            optimizer.step()
            save_loss(loss.item(), args.path_to_model_save /
                      f"loss_{model}_{args.learning_rate}_{args.epochs}.txt".replace("\n", ""))
            print(f"batch {batch_number} done")
        scheduler.step()
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

        # Compute weighted metrics
        weighted_precision = precision_score(
            y_true, y_pred, average="weighted")
        weighted_recall = recall_score(y_true, y_pred, average="weighted")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

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
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
            "Weighted F1 Score": weighted_f1,
            "ROC AUC": roc_auc
        }

        # Print results before saving
        print("Test Results:")
        for key, value in results.items():
            print(f"{key}: {value}")

        write_dict_to_csv(results, 'results.csv', write_headers=True)
        print('result saved in the file results.csv')
    except UndefinedMetricWarning as umw:
        print(f"Warning: {umw}")


def evaluate(args, model, device) -> None:
    _, _, test_loader = init(args)
    y_predictions = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch_number, data in enumerate(test_loader):
            print(f"start batch {batch_number}")
            try:
                X, y = data
                X_rgb = X.to(device)
                X_f = X.to(device)
                y = y.to(device)

                y_true.append(y.cpu().numpy())
                y_pred = model(X_rgb, X_f)
                y_pred = y_pred.argmax(dim=1).detach().cpu().numpy()
                y_predictions.append(y_pred)

                print(f"end batch {batch_number}")
            except Exception as e:
                print(f"Error in batch {batch_number}: {e}")
                continue

    y_true = np.concatenate(y_true, axis=0)
    y_predictions = np.concatenate(y_predictions, axis=0)

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
    parser.add_argument("--test_percentage", type=float,
                        default=20.0, help="Percentage of data to use for testing")
    parser.add_argument("--downsample", action="store_true",
                        help="Downsample the dataset to balance classes")
    parser.add_argument("--downsample_size", type=int, default=None,
                        help="Target size for downsampling classes")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(
        args, device=device, n_epochs=args.epochs
    )
    save_model_weights(model, args.path_to_model_save /
                       f"{model}_{args.learning_rate}_{args.epochs}.pth")
    evaluate(args, model, device)


if __name__ == "__main__":
    main()
