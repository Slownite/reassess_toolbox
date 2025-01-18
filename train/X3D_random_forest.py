from argparse import ArgumentParser
import pathlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    fbeta_score
)
from sklearn.exceptions import UndefinedMetricWarning
from datasets import MultiNpyEdf
from utils import write_dict_to_csv, downsample


def load_data(dataset, dataset_path, schema_json, downsample_classes=False, downsample_seed=0):
    """Load data using the MultiNpyEdf dataset class, with optional downsampling."""
    npy_files = dataset_path.rglob("0rgb_*x3d*.npy")
    edf_files = dataset_path.rglob("*.edf")
    data = dataset(npy_files, edf_files, schema_json)

    if downsample_classes:
        print("Downsampling the dataset for balanced classes...")
        data = downsample(data, seed=downsample_seed)

    X, y = [], []
    for idx in range(len(data)):
        try:
            features, label = data[idx]
            X.append(features.flatten())  # Flatten features for Random Forest
            y.append(label)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            continue

    return np.array(X), np.array(y)


def train_random_forest(X_train, y_train, n_estimators=100):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true, y_pred):
    """Compute and print metrics for the model."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        f2 = fbeta_score(y_true, y_pred, beta=2)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(y_true, y_pred)

        results = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score": f2,
            "Specificity": specificity,
            "ROC AUC": roc_auc,
        }

        # Print results
        for metric, value in results.items():
            print(f"{metric}: {value}")

        return results

    except UndefinedMetricWarning as umw:
        print(f"Warning: {umw}")
        return {}


def main():
    parser = ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("schema_path", type=pathlib.Path)
    parser.add_argument("--testset", type=pathlib.Path, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--downsample", action="store_true",
                        help="Enable downsampling for class balance.")
    parser.add_argument("--downsample_seed", type=int,
                        default=0, help="Random seed for downsampling.")
    args = parser.parse_args()

    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data(
        MultiNpyEdf,
        args.data_path,
        args.schema_path,
        downsample_classes=args.downsample,
        downsample_seed=args.downsample_seed
    )

    # Train Random Forest model
    print("Training Random Forest...")
    model = train_random_forest(
        X_train, y_train, n_estimators=args.n_estimators)

    # Evaluate on the test set if provided
    if args.testset:
        print("Loading test data...")
        X_test, y_test = load_data(
            MultiNpyEdf,
            args.testset,
            args.schema_path,
            downsample_classes=args.downsample,
            downsample_seed=args.downsample_seed
        )

        print("Evaluating Random Forest...")
        y_pred = model.predict(X_test)

        # Compute metrics
        results = compute_metrics(y_test, y_pred)

        # Save metrics to CSV
        write_dict_to_csv(
            results, 'random_forest_results.csv', write_headers=True)
        print("Results saved to random_forest_results.csv")


if __name__ == "__main__":
    main()
