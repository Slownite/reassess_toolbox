#!/usr/bin/env python3

import pathlib
import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    fbeta_score
)

logging.basicConfig(level=logging.INFO)


def load_data(dataset_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a CSV file or similar format. Assumes the last column is the target.
    """
    logging.info(f"Loading data from {dataset_path}.")
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values  # Features (all columns except the last one)
    y = data.iloc[:, -1].values   # Target (last column)
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute evaluation metrics for binary classification.
    """
    metrics = {}

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

    # F-beta score for prioritizing recall or precision
    metrics['fbeta_0.5'] = fbeta_score(
        y_true, y_pred, beta=0.5, zero_division=0)
    metrics['fbeta_2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    return metrics


def train(args) -> xgb.Booster:
    """
    Train an XGBoost model using GPU.
    """
    logging.info("Loading and preparing data...")
    X, y = load_data(args.data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    logging.info("Converting data to DMatrix format for XGBoost.")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "device": "cuda",
        "tree_method": "hist",  # Use GPU for training
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "nthread": args.n_jobs,
    }

    logging.info("Starting training...")
    evals = [(dtrain, "train"), (dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds
    )

    logging.info("Training complete.")
    return model


def evaluate(args, model) -> None:
    """
    Evaluate the model on the test dataset and compute metrics.
    """
    logging.info("Loading test data...")
    X_test, y_test = load_data(args.testset)
    dtest = xgb.DMatrix(X_test, label=y_test)

    logging.info("Making predictions...")
    y_pred_proba = model.predict(dtest)
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)

    logging.info("Computing metrics...")
    metrics = compute_metrics(y_test, y_pred)
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = args.output_dir / "evaluation_metrics.csv"

    logging.info(f"Saving metrics to {metrics_csv_path}.")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(metrics)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path,
                        help="Path to the training dataset.")
    parser.add_argument("--testset", type=pathlib.Path,
                        help="Path to the test dataset.", required=True)
    parser.add_argument("--output_dir", type=pathlib.Path,
                        default=pathlib.Path("./output"), help="Directory to save outputs.")
    parser.add_argument("--learning_rate", type=float,
                        default=0.1, help="Learning rate.")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum depth of trees.")
    parser.add_argument("--min_child_weight", type=float, default=1.0,
                        help="Minimum sum of weights of all children.")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio of training instances.")
    parser.add_argument("--colsample_bytree", type=float,
                        default=0.8, help="Subsample ratio of columns.")
    parser.add_argument("--num_boost_round", type=int,
                        default=100, help="Number of boosting rounds.")
    parser.add_argument("--early_stopping_rounds", type=int,
                        default=10, help="Early stopping rounds.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use as validation set.")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of parallel threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    model = train(args)

    # Save the model
    model_path = args.output_dir / "xgboost_model.bst"
    logging.info(f"Saving model to {model_path}.")
    model.save_model(model_path)

    # Evaluate the model
    evaluate(args, model)


if __name__ == "__main__":
    main()
