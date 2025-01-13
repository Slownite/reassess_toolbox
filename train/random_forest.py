import pathlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import joblib
import argparse
import json


class I3DLoader:
    def __init__(self, dataset_path, schema_path, policy="two_class_policy"):
        from utils import process_annotation_text_file
        from more_itertools import flatten

        self.dataset_path = pathlib.Path(dataset_path)
        self.schema_path = pathlib.Path(schema_path)

        with open(self.schema_path, 'r') as f:
            self.schema = json.load(f)

        policies = {
            "two_class_policy": lambda x: 1 if 1 in x else 0
        }

        all_files = sorted(self.dataset_path.glob("**/*"))
        rgb_npy_files = [f for f in all_files if f.suffix ==
                         '.npy' and f.name.startswith("rgb_")]
        flow_npy_files = [f for f in all_files if f.suffix ==
                          '.npy' and f.name.startswith("flow_")]
        txt_files = [f for f in all_files if f.suffix == '.txt']

        self.annotations = list(flatten([process_annotation_text_file(
            path, self.schema, policies[policy]
        ) for path in txt_files]))

        rgb_list = [np.load(npy_path) for npy_path in rgb_npy_files]
        flow_list = [np.load(npy_path) for npy_path in flow_npy_files]

        self.rgb_tensors = np.concatenate(rgb_list, axis=0).squeeze()
        self.flow_tensors = np.concatenate(flow_list, axis=0).squeeze()

        # Standardize lengths
        self.rgb_tensors, self.flow_tensors = self.standardize(
            self.rgb_tensors, self.flow_tensors
        )

    def standardize(self, rgb, flow):
        if rgb.shape[0] > flow.shape[0]:
            new_rgb = rgb[:len(flow)]
            return new_rgb, flow
        elif rgb.shape[0] < flow.shape[0]:
            new_flow = flow[:len(rgb)]
            return rgb, new_flow
        else:
            return rgb, flow

    def get_data(self):
        X = np.concatenate([
            self.rgb_tensors.reshape(len(self.rgb_tensors), -1),
            self.flow_tensors.reshape(len(self.flow_tensors), -1)
        ], axis=1)
        y = np.array(self.annotations)

        return X, y


def train_random_forest_with_smote(X, y, model_save_path, n_estimators=100, n_jobs=-1):
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(
        f"Data after SMOTE: Features: {X_resampled.shape}, Labels: {y_resampled.shape}")

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
    rf_model.fit(X_resampled, y_resampled)

    # Save model
    joblib.dump(rf_model, model_save_path)
    print(f"Random Forest model saved to {model_save_path}")


def evaluate_random_forest(model_path, test_dataset, test_schema, metrics_save_path):
    # Load model
    rf_model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Load test data
    test_loader = I3DLoader(test_dataset, test_schema)
    X_test, y_test = test_loader.get_data()
    print(f"Test Data Shape: Features: {X_test.shape}, Labels: {y_test.shape}")

    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics
    with open(metrics_save_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Metrics saved to {metrics_save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset", type=str,
                        help="Path to the training dataset directory")
    parser.add_argument("train_schema", type=str,
                        help="Path to the training schema JSON file")
    parser.add_argument("model_save_path", type=str,
                        help="Path to save the trained Random Forest model")
    parser.add_argument("--test_dataset", type=str,
                        help="Path to the test dataset directory", default=None)
    parser.add_argument("--test_schema", type=str,
                        help="Path to the test schema JSON file", default=None)
    parser.add_argument("--metrics_save_path", type=str,
                        help="Path to save evaluation metrics", default="metrics.json")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the Random Forest (default: 100)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of CPU cores to use for training (default: -1, use all available cores)")
    args = parser.parse_args()

    # Load training data
    train_loader = I3DLoader(args.train_dataset, args.train_schema)
    X_train, y_train = train_loader.get_data()

    print(
        f"Training Data Shape: Features: {X_train.shape}, Labels: {y_train.shape}")

    # Train Random Forest with SMOTE
    # train_random_forest_with_smote(X_train, y_train, args.model_save_path,
    #                                n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    #
    # # Evaluate if test dataset is provided
    # if args.test_dataset and args.test_schema:
    #     evaluate_random_forest(
    #         args.model_save_path, args.test_dataset, args.test_schema, args.metrics_save_path)


if __name__ == "__main__":
    main()
