from datasets import I3D_dataset
from modules import I3D
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import argparse
from tqdm.auto import tqdm
# Define the training function


def train_I3D(dataset_path, annotation_schema_path, model, num_epochs=10, batch_size=32, learning_rate=0.001, input_type="rgb"):
    """
    Training function for I3D model.

    Parameters:
    - dataset_path: Path to the dataset
    - annotation_schema_path: Path to the annotation schema
    - model: The PyTorch model (e.g., I3D)
    - num_epochs: Number of training epochs
    - batch_size: Size of training batch
    - learning_rate: Learning rate for optimizer
    - input_type: Type of input ('rgb' or 'flow')
    """
    if input_type not in ["rgb", "flow"]:
        raise ValueError("input_type must be 'rgb' or 'flow'")

    # Initialize dataset and DataLoader
    dataset = I3D_dataset(
        path=dataset_path, annotation_schema_path=annotation_schema_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Inputs and labels to device
            # Select rgb or flow
            inputs = inputs[0] if input_type == "rgb" else inputs[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)  # Model takes either RGB or flow as input
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # Epoch results
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch [{
              epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training complete.")

# Define the testing function


def test_I3D(dataset_path, annotation_schema_path, model, batch_size=32, input_type="rgb"):
    """
    Testing function for I3D model.

    Parameters:
    - dataset_path: Path to the dataset
    - annotation_schema_path: Path to the annotation schema
    - model: The PyTorch model (e.g., I3D)
    - batch_size: Size of testing batch
    - input_type: Type of input ('rgb' or 'flow')
    """
    if input_type not in ["rgb", "flow"]:
        raise ValueError("input_type must be 'rgb' or 'flow'")

    # Initialize dataset and DataLoader
    dataset = I3D_dataset(
        path=dataset_path, annotation_schema_path=annotation_schema_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            # Inputs and labels to device
            # Select rgb or flow
            inputs = inputs[0] if input_type == "rgb" else inputs[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # Assuming binary classification
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class="ovr") if len(
        set(all_labels)) > 2 else roc_auc_score(all_labels, all_probabilities)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train or Test the I3D model.")
    parser.add_argument(
        "mode", choices=["train", "test"], help="Mode: train or test.")
    parser.add_argument("--dataset_path", type=pathlib.Path,
                        required=True, help="Path to the dataset.")
    parser.add_argument("--annotation_schema_path", type=str,
                        required=True, help="Path to the annotation schema.")
    parser.add_argument("--model_path", type=str,
                        required=True, help="Path to the model file.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training/testing.")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001, help="Learning rate for training.")
    parser.add_argument(
        "--input_type", choices=["rgb", "flow"], default="rgb", help="Type of input data.")

    args = parser.parse_args()

    # Load the model
    model = I3D(num_classes=2, pretrained_weights=args.model_path)

    if args.mode == "train":
        train_I3D(
            dataset_path=args.dataset_path,
            annotation_schema_path=args.annotation_schema_path,
            model=model,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            input_type="rgb"
        )
    elif args.mode == "test":
        test_I3D(
            dataset_path=args.dataset_path,
            annotation_schema_path=args.annotation_schema_path,
            model=model,
            batch_size=args.batch_size,
            input_type="rgb"
        )


if __name__ == "__main__":
    main()
